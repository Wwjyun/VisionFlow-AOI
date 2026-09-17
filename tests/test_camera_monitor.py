from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from core.camera_monitor_processor import CameraFrameQueue, CameraMonitorProcessor, CapturedFrame
from core.image_loader import ImageLoadError, frame_to_bgr, load_image
from core.pipeline import AOIPipeline
from devices.ccd_models import (
    CameraConnectionSettings,
    CameraRecipeSettings,
    ImageSaveFormat,
    TriggerMode,
    TriggerSettings,
)
from devices.ccd_settings_store import CcdMachineSettingsStore
from devices.factory import CcdDevices
from devices.frame_writer import write_frame_atomic
from devices.simulated import SimulatedLineScanCamera, SimulatedMeterWheel
from gui.ccd_controller import CcdController
from gui.main_window import CAMERA_MONITOR_NO_ORIGINAL_MESSAGE, CAMERA_MONITOR_READY_MESSAGE, MainWindow
from gui.workers import CameraMonitorWorker
from gui_launcher import _packaged_smoke_recipe


def _wait_until(predicate, timeout: float = 10.0) -> bool:
    app = QApplication.instance()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if app is not None:
            app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _gray_frame(seed: int = 20260917, shape=(128, 160)) -> np.ndarray:
    frame = np.random.default_rng(seed).integers(0, 256, size=shape, dtype=np.uint8)
    frame.setflags(write=False)
    return frame


def _write_recipe(root: Path, save_json: bool = True) -> Path:
    recipe = _packaged_smoke_recipe()
    recipe["output"]["save_json"] = save_json
    path = root / "camera_smoke.yaml"
    path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")
    return path


def _comparable(result: dict) -> dict:
    normalized = deepcopy(result)
    for key in ("duration_sec", "outputs", "execution", "provenance", "image_name", "source"):
        normalized.pop(key, None)
    for tile_result in normalized["tiles"]:
        for detector_result in tile_result["detectors"]:
            detector_result.pop("execution", None)
    return normalized


class FrameConversionTests(unittest.TestCase):
    def test_gray_frame_matches_the_same_pixels_saved_as_8_bit_bmp(self):
        frame = _gray_frame()
        with tempfile.TemporaryDirectory() as directory:
            path = write_frame_atomic(frame, Path(directory) / "相機.bmp", ImageSaveFormat.BMP)
            from_file = load_image(path)
        converted = frame_to_bgr(frame)
        np.testing.assert_array_equal(converted, from_file)
        self.assertEqual(converted.shape, (128, 160, 3))
        self.assertTrue(converted.flags.writeable)
        self.assertFalse(np.shares_memory(converted, frame))

    def test_color_frames_are_copied_and_invalid_frames_are_rejected(self):
        color = np.zeros((4, 5, 3), dtype=np.uint8)
        color.setflags(write=False)
        copied = frame_to_bgr(color)
        self.assertTrue(copied.flags.writeable)
        self.assertFalse(np.shares_memory(copied, color))
        for invalid in (np.zeros((4, 5), dtype=np.uint16), np.zeros((4, 5, 4), dtype=np.uint8), np.zeros(8, dtype=np.uint8)):
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype), self.assertRaises(ImageLoadError):
                frame_to_bgr(invalid)


class PipelineFrameTests(unittest.TestCase):
    def test_run_frame_matches_file_inspection_and_records_the_source(self):
        frame = _gray_frame()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe_path = _write_recipe(root)
            bmp_path = write_frame_atomic(frame, root / "frame.bmp", ImageSaveFormat.BMP)
            from_file = AOIPipeline(recipe_path, root / "file").run(bmp_path)
            metadata = {"frame_index": 7, "trigger_mode": "software_trigger"}
            from_frame = AOIPipeline(recipe_path, root / "frame").run_frame(frame, "camera_20260917_000007", metadata)
            json_payload = json.loads(Path(from_frame["outputs"]["json"]).read_text(encoding="utf-8"))

        self.assertGreater(from_file["summary"]["defect_count"], 0, "the fixture must exercise real defects")
        self.assertEqual(_comparable(from_frame), _comparable(from_file))
        self.assertEqual(from_frame["image_name"], "camera_20260917_000007")
        self.assertEqual(from_frame["source"], {"type": "camera", **metadata})
        self.assertNotIn("source", from_file, "file inspections keep their existing result schema")
        self.assertEqual(json_payload["source"]["frame_index"], 7)
        self.assertTrue(Path(from_frame["outputs"]["json"]).name.startswith("camera_20260917_000007"))


class FrameQueueTests(unittest.TestCase):
    def test_queue_is_bounded_records_drops_and_rejects_after_close(self):
        queue = CameraFrameQueue(capacity=2)
        frames = [CapturedFrame(_gray_frame(index, (2, 2)), f"f{index}", float(index), {"i": index}) for index in range(4)]
        self.assertEqual([queue.put(frame) for frame in frames], [True, True, False, False])
        self.assertEqual([dropped.source_name for dropped in queue.take_dropped()], ["f2", "f3"])
        self.assertEqual(queue.take_dropped(), [])
        self.assertEqual(queue.get(0).source_name, "f0")
        queue.close()
        self.assertFalse(queue.put(frames[0]))
        self.assertEqual(queue.get(0).source_name, "f1", "frames queued before close are still delivered")
        self.assertIsNone(queue.get(0.01))


class CameraMonitorProcessorTests(unittest.TestCase):
    def test_frames_are_inspected_in_order_drops_are_reported_and_queue_drains_on_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe_path = _write_recipe(root, save_json=False)
            queue = CameraFrameQueue(capacity=2)
            for index in range(3):
                queue.put(CapturedFrame(_gray_frame(index), f"camera_{index}", time.perf_counter(), {"frame_index": index}))
            items: list[dict] = []
            processor = CameraMonitorProcessor(
                queue,
                recipe_path,
                root / "out",
                item_callback=items.append,
                stop_callback=lambda: any(item["final_result"] != "ERROR" for item in items),
            )
            summary = processor.run()

        self.assertEqual([item["image_name"] for item in items], ["camera_2", "camera_0", "camera_1"])
        dropped = items[0]
        self.assertEqual(dropped["final_result"], "ERROR")
        self.assertIn("檢測佇列已滿", dropped["error"])
        inspected = items[1:]
        self.assertTrue(all(item["final_result"] in {"PASS", "NG"} for item in inspected))
        self.assertEqual([item["camera"]["frame_index"] for item in inspected], [0, 1])
        self.assertTrue(all(item["source"] == "camera" for item in items))
        self.assertEqual(set(inspected[0]["timing"]), {"queue_wait_sec", "pipeline_and_reports_sec", "end_to_end_sec"})
        self.assertEqual(inspected[0]["duration_sec"], inspected[0]["timing"]["end_to_end_sec"])
        self.assertEqual(inspected[0]["detail"]["source"]["type"], "camera")
        self.assertEqual((summary["processed"], summary["dropped"], summary["source"]), (2, 1, "camera"))
        self.assertTrue(queue.closed)

    def test_worker_failure_is_reported_and_closes_the_queue(self):
        app = QApplication.instance() or QApplication([])
        queue = CameraFrameQueue()
        with tempfile.TemporaryDirectory() as directory:
            worker = CameraMonitorWorker(queue, Path(directory) / "missing.yaml", Path(directory))
            failures: list[str] = []
            worker.failed.connect(failures.append)
            worker.run()
        app.processEvents()
        self.assertEqual(len(failures), 1)
        self.assertTrue(queue.closed)


class ControllerHandOffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.camera = SimulatedLineScanCamera(width=16, auto_emit=False)
        self.controller = CcdController(
            CcdDevices(self.camera, SimulatedMeterWheel()),
            CcdMachineSettingsStore(Path(self._temp.name) / "ccd.json"),
        )

    def tearDown(self):
        self.controller.close()
        self._temp.cleanup()

    def _connect(self, mode: TriggerMode) -> None:
        self.controller.apply_camera_settings(
            CameraConnectionSettings(), CameraRecipeSettings(trigger=TriggerSettings(mode))
        )
        self.controller.connect_camera()

    def test_only_trigger_frames_are_handed_to_inspection(self):
        self.assertIn("相機未連線", self.controller.camera_monitor_blocker())
        queue = CameraFrameQueue()
        self._connect(TriggerMode.CONTINUOUS)
        self.assertIn("連續取像", self.controller.camera_monitor_blocker())
        self.controller.attach_inspection_queue(queue)
        self.camera.emit_frame()
        self.assertEqual(queue.pending(), 0)

        self.controller.disconnect_camera()
        self._connect(TriggerMode.EXTERNAL)
        self.assertEqual(self.controller.camera_monitor_blocker(), "")
        self.controller.attach_inspection_queue(queue)
        first = self.camera.emit_frame()
        self.camera.emit_frame()
        self.controller.detach_inspection_queue()
        self.camera.emit_frame()
        self.assertEqual(queue.pending(), 2)
        captured = [queue.get(0), queue.get(0)]
        self.assertIs(captured[0].image, first, "the read-only driver frame is handed over without a copy")
        self.assertEqual([frame.metadata["frame_index"] for frame in captured], [1, 2])
        self.assertEqual(captured[0].metadata["trigger_mode"], "external_trigger")
        self.assertEqual((captured[0].metadata["frame_width"], captured[0].metadata["frame_height"]), (16, 720))
        self.assertLess(captured[0].source_name, captured[1].source_name)
        self.assertTrue(captured[0].source_name.startswith("camera_"))


class MainWindowCameraMonitorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_camera_source_inspects_trigger_frames_until_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera = SimulatedLineScanCamera(width=96, max_frame_height=96, auto_emit=False)
            window = MainWindow(
                settings=QSettings(str(root / "gui.ini"), QSettings.Format.IniFormat),
                ccd_devices=CcdDevices(camera, SimulatedMeterWheel()),
                ccd_settings_store=CcdMachineSettingsStore(root / "ccd.json"),
            )
            try:
                window.permission_manager.switch_mode("eng", "1234")
                window.mode = "eng"
                window._apply_mode_permissions()
                window.output_dir = str(root / "outputs")
                window._load_recipe(_write_recipe(root, save_json=False))
                window._on_monitor_source_changed("camera")
                panel = window.monitor_screen.control_panel
                self.assertIn("相機未連線", panel.message_label.text())

                window.ccd_controller.apply_camera_settings(
                    CameraConnectionSettings(), CameraRecipeSettings(trigger=TriggerSettings(TriggerMode.EXTERNAL))
                )
                window.ccd_controller.connect_camera()
                self.assertEqual(panel.message_label.text(), CAMERA_MONITOR_READY_MESSAGE)
                self.assertTrue(panel.start_button.isEnabled())

                window._start_monitoring()
                self.assertTrue(window.monitor_running)
                self.assertFalse(panel.source_segmented.isEnabled())
                camera.emit_frame()
                camera.emit_frame()
                self.assertTrue(_wait_until(lambda: len(window.monitor_screen.items()) == 2))
                items = window.monitor_screen.items()
                self.assertTrue(all(item["source"] == "camera" for item in items))
                self.assertTrue(all(item["final_result"] in {"PASS", "NG"} for item in items))

                window._stop_monitoring()
                self.assertIsNone(window.ccd_controller._inspection_queue)
                camera.emit_frame()
                self.assertTrue(_wait_until(lambda: not window.monitor_running))
                self.assertEqual(len(window.monitor_screen.items()), 2, "frames after Stop are not inspected")
                self.assertEqual(window.monitor_result["processed"], 2)
                self.assertTrue(Path(window.monitor_result["output_dir"]).name.endswith("_camera"))

                window._open_monitor_original_image(items[0])
                self.assertEqual(window.notice_bar.label.text(), CAMERA_MONITOR_NO_ORIGINAL_MESSAGE)
            finally:
                window.ccd_controller.close()
                window._inspection_gpu_sessions.close()
                window.deleteLater()


if __name__ == "__main__":
    unittest.main()
