from __future__ import annotations

import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from core.recipe_manager import RecipeError, RecipeManager
from devices.ccd_models import AcquisitionSettings, CameraRecipeSettings, TriggerMode, TriggerSettings
from devices.ccd_recipe import camera_section, camera_settings_from_recipe, parse_camera_section
from devices.ccd_settings_store import CcdMachineSettingsStore
from devices.factory import CcdDevices
from devices.simulated import SimulatedLineScanCamera, SimulatedMeterWheel
from gui.designer_model import DesignerRecipeMapper, RecipeDraft
from gui.main_window import MainWindow
from gui.screens.designer_screen import DesignerScreen

ROOT = Path(__file__).resolve().parents[1]
BASE_RECIPE = ROOT / "recipes" / "PRODUCT_A_AOI_01.yaml"

CAMERA_SECTION = {
    "exposure_time": 1234.56,
    "gain": 2.345,
    "length_lines": 16384,
    "internal_line_rate_hz": 5000,
    "trigger": {
        "mode": "external_trigger",
        "external_frame_one_frame": True,
        "compare_follows_encoder": True,
        "set_encoder_on_trigger": False,
    },
    "auto_save": {"external_one_frame": True, "software_trigger": False},
}


def _recipe_with_camera(camera: dict | None = CAMERA_SECTION) -> dict:
    recipe = RecipeManager().load(BASE_RECIPE)
    if camera is not None:
        recipe["camera"] = deepcopy(camera)
    return recipe


def _type_into(stepper, value) -> None:
    """Edit a NumStepper the way an operator does: type, then finish editing."""
    stepper.edit.setText(str(value))
    stepper.edit.editingFinished.emit()


class CameraRecipeCodecTests(unittest.TestCase):
    def test_section_round_trips_through_yaml_without_changing_values(self):
        settings = parse_camera_section(CAMERA_SECTION)
        self.assertEqual(settings.acquisition, AcquisitionSettings(1234.56, 2.345, 16384, 5000))
        self.assertEqual(settings.trigger, TriggerSettings(TriggerMode.EXTERNAL, True, True, False))
        self.assertTrue(settings.auto_save_external_one_frame)
        dumped = yaml.safe_load(yaml.safe_dump(camera_section(settings), allow_unicode=True, sort_keys=False))
        self.assertEqual(dumped, CAMERA_SECTION)
        self.assertEqual(parse_camera_section(dumped), settings)

    def test_absent_or_null_section_means_no_camera_settings(self):
        self.assertIsNone(camera_settings_from_recipe({}))
        self.assertIsNone(camera_settings_from_recipe({"camera": None}))
        self.assertIsNone(camera_settings_from_recipe(None))
        minimal = {key: CAMERA_SECTION[key] for key in ("exposure_time", "gain", "length_lines", "internal_line_rate_hz")}
        minimal["trigger"] = {"mode": "continuous"}
        settings = parse_camera_section(minimal)
        self.assertEqual(settings.trigger, TriggerSettings())
        self.assertFalse(settings.auto_save_software_trigger)

    def test_recipe_manager_rejects_invalid_camera_sections(self):
        manager = RecipeManager()
        manager.validate(_recipe_with_camera())
        manager.validate(_recipe_with_camera(None))

        def changed(**updates):
            section = deepcopy(CAMERA_SECTION)
            for dotted, value in updates.items():
                target = section
                *parents, key = dotted.split("__")
                for parent in parents:
                    target = target[parent]
                if value is KeyError:
                    del target[key]
                else:
                    target[key] = value
            return section

        cases = {
            "not a mapping": ("camera must be a mapping", ["exposure"]),
            "unknown key": ("unknown keys: brightness", changed(brightness=1)),
            "missing exposure": ("exposure_time is required", changed(exposure_time=KeyError)),
            "boolean gain": ("gain must be a number", changed(gain=True)),
            "float length": ("length_lines must be an integer", changed(length_lines=720.0)),
            "out of range": ("gain must be between", changed(gain=1000.5)),
            "not finite": ("exposure_time must be a number", changed(exposure_time=float("nan"))),
            "bad mode": ("trigger.mode must be one of", changed(trigger__mode="free_run")),
            "missing trigger": ("camera.trigger must be a mapping", changed(trigger=KeyError)),
            "unknown trigger key": ("trigger has unknown keys", changed(trigger__source="line")),
            "string flag": ("must be true or false", changed(trigger__external_frame_one_frame="yes")),
            "software one frame": (
                "not allowed with mode software_trigger: external_frame_one_frame, compare_follows_encoder",
                changed(trigger__mode="software_trigger"),
            ),
            "encoder without compare": (
                "set_encoder_on_trigger",
                changed(trigger__compare_follows_encoder=False, trigger__set_encoder_on_trigger=True),
            ),
            "auto save type": ("auto_save.software_trigger must be true or false", changed(auto_save__software_trigger=1)),
        }
        for name, (message, section) in cases.items():
            with self.subTest(name):
                with self.assertRaises(RecipeError) as raised:
                    manager.validate(_recipe_with_camera(section))
                self.assertIn(message, str(raised.exception))

    def test_tracked_recipes_have_no_camera_section_and_still_load(self):
        for path in sorted((ROOT / "recipes").glob("*.yaml")):
            with self.subTest(path.name):
                recipe = RecipeManager().load(path)
                self.assertNotIn("camera", recipe)

    def test_designer_mapper_only_writes_camera_when_present(self):
        draft = RecipeDraft("R", "P", "M", "1", {"mode": "grid"}, {"mode": "cpu"}, {"900-CS-AP-1": {}}, None)
        self.assertNotIn("camera", DesignerRecipeMapper.build(draft))
        recipe = DesignerRecipeMapper.build(
            RecipeDraft("R", "P", "M", "1", {"mode": "grid"}, {"mode": "cpu"}, {}, None, camera=CAMERA_SECTION)
        )
        self.assertEqual(recipe["camera"], CAMERA_SECTION)


class DesignerCameraPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_engineer_mode_preserves_camera_section_exactly_and_cannot_edit(self):
        designer = DesignerScreen()
        designer.set_mode("eng")
        designer.set_recipe(_recipe_with_camera())
        panel = designer.camera_panel
        self.assertFalse(designer.is_dirty())
        self.assertFalse(panel.include_toggle.isEnabled())
        self.assertFalse(panel.exposure_input.isEnabled())
        self.assertEqual(designer.build_recipe()["camera"], CAMERA_SECTION)

        designer.set_recipe(_recipe_with_camera(None))
        self.assertNotIn("camera", designer.build_recipe())
        self.assertFalse(designer.is_dirty())

    def test_admin_edits_are_dirty_follow_trigger_rules_and_validate(self):
        designer = DesignerScreen()
        designer.set_mode("admin")
        designer.set_recipe(_recipe_with_camera())
        panel = designer.camera_panel
        self.assertTrue(panel.exposure_input.isEnabled())

        _type_into(panel.exposure_input, 900.0)
        self.assertTrue(designer.is_dirty())
        combo = panel.trigger_mode_combo
        combo.setCurrentIndex(combo.findData(TriggerMode.SOFTWARE.value))
        self.assertFalse(panel.one_frame_toggle.isChecked())
        self.assertFalse(panel.one_frame_toggle.isEnabled())
        self.assertFalse(panel.compare_follow_toggle.isChecked())
        self.assertTrue(panel.auto_save_software_toggle.isEnabled())
        panel.auto_save_software_toggle.setChecked(True)

        recipe = designer.build_recipe()
        designer._recipe_validator.validate(recipe)
        camera = recipe["camera"]
        self.assertEqual(camera["exposure_time"], 900.0)
        self.assertEqual(camera["trigger"]["mode"], "software_trigger")
        self.assertFalse(camera["trigger"]["external_frame_one_frame"])
        self.assertTrue(camera["auto_save"]["software_trigger"])
        self.assertEqual(camera["length_lines"], 16384)

        panel.include_toggle.setChecked(False)
        self.assertFalse(panel.exposure_input.isEnabled())
        self.assertNotIn("camera", designer.build_recipe())

    def test_admin_can_add_camera_section_to_a_legacy_recipe(self):
        designer = DesignerScreen()
        designer.set_mode("admin")
        designer.set_recipe(_recipe_with_camera(None))
        designer.camera_panel.include_toggle.setChecked(True)
        self.assertTrue(designer.is_dirty())
        recipe = designer.build_recipe()
        RecipeManager().validate(recipe)
        self.assertEqual(parse_camera_section(recipe["camera"]), CameraRecipeSettings())

    def test_ccd_settings_become_an_unsaved_designer_edit_once(self):
        designer = DesignerScreen()
        designer.set_mode("eng")
        designer.set_recipe(_recipe_with_camera(None))
        settings = CameraRecipeSettings(AcquisitionSettings(800, 3, 2000, 100))
        self.assertTrue(designer.apply_camera_settings(settings))
        self.assertTrue(designer.is_dirty())
        self.assertEqual(parse_camera_section(designer.build_recipe()["camera"]), settings.normalized())
        self.assertFalse(designer.apply_camera_settings(settings))


class MainWindowCameraRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.camera = SimulatedLineScanCamera(width=16, auto_emit=False)
        self.window = MainWindow(
            settings=QSettings(str(self.root / "gui.ini"), QSettings.Format.IniFormat),
            ccd_devices=CcdDevices(self.camera, SimulatedMeterWheel()),
            ccd_settings_store=CcdMachineSettingsStore(self.root / "ccd.json"),
        )
        self.addCleanup(self.window.deleteLater)
        self.addCleanup(self.window._inspection_gpu_sessions.close)
        self.addCleanup(self.window.ccd_controller.close)
        self.window.permission_manager.switch_mode("admin", "5678")
        self.window.mode = "admin"
        self.window._apply_mode_permissions()

    def tearDown(self):
        self._temp.cleanup()

    def _write_recipe(self, name: str, camera: dict | None) -> Path:
        path = self.root / name
        path.write_text(yaml.safe_dump(_recipe_with_camera(camera), allow_unicode=True, sort_keys=False), encoding="utf-8")
        return path

    def test_loading_recipes_applies_camera_sections_and_keeps_settings_for_legacy_recipes(self):
        controller = self.window.ccd_controller
        screen = self.window.ccd_screen
        self.window._load_recipe(self._write_recipe("相機產品.yaml", CAMERA_SECTION))
        expected = parse_camera_section(CAMERA_SECTION)
        self.assertEqual(controller.product_settings, expected)
        self.assertEqual(screen.length_input.value(), 16384)
        self.assertTrue(screen.auto_save_external_check.isChecked())
        self.assertIn("相機產品.yaml", screen.product_source_label.text())
        self.assertIn("外部觸發", self.window.monitor_screen.control_panel.camera_status_label.text())
        self.assertEqual(screen.exposure_input.value(), 1234.6, "the stepper only displays a rounded value")
        screen.apply_camera_button.click()
        self.assertEqual(controller.product_settings, expected, "applying unedited fields must not round them")
        self.assertFalse(self.window.designer_screen.is_dirty())

        self.window._load_recipe(self._write_recipe("legacy.yaml", None))
        self.assertEqual(controller.product_settings, expected, "a Recipe without camera must not change the camera")
        self.assertIn("未包含相機設定", screen.product_source_label.text())

        controller.connect_camera()
        changed = deepcopy(CAMERA_SECTION)
        changed["length_lines"] = 2048
        self.window._load_recipe(self._write_recipe("changed.yaml", changed))
        self.assertTrue(controller.pending_hardware_write())
        self.assertIn("需斷線重連", self.window.notice_bar.label.text())
        self.assertFalse(screen.pending_label.isHidden())

    def test_ccd_apply_syncs_to_designer_and_saved_recipe_reloads_into_the_session(self):
        screen = self.window.ccd_screen
        _type_into(screen.length_input, 3000)
        screen.apply_camera_button.click()
        self.assertIn("未載入 Recipe", self.window.notice_bar.label.text())
        self.assertFalse(self.window.designer_screen.is_dirty())

        self.window._load_recipe(self._write_recipe("legacy.yaml", None))
        self.assertFalse(self.window.designer_screen.is_dirty())
        _type_into(screen.exposure_input, 777)
        screen.apply_camera_button.click()
        self.assertTrue(self.window.designer_screen.is_dirty())
        self.assertIn("已同步到 Recipe 設計", self.window.notice_bar.label.text())
        self.assertIn("尚未儲存到 Recipe", screen.product_source_label.text())
        section = self.window.designer_screen.build_recipe()["camera"]
        self.assertEqual((section["exposure_time"], section["length_lines"]), (777.0, 3000))

        saved_path = self.root / "saved.yaml"
        with patch("gui.screens.designer_screen.QFileDialog.getSaveFileName", return_value=(str(saved_path), "")):
            self.window.designer_screen._save_recipe()
        self.assertFalse(self.window.designer_screen.is_dirty())
        on_disk = yaml.safe_load(saved_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["camera"]["exposure_time"], 777.0)
        self.assertEqual(self.window.recipe_path, saved_path)
        self.assertEqual(screen.product_source_label.text(), "來源：Recipe「saved.yaml」。")


if __name__ == "__main__":
    unittest.main()
