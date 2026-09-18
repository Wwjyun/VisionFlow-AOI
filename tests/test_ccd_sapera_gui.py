from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from devices.ccd_models import (
    TRIGGER_MODE_LABELS,
    AcquisitionSettings,
    CameraConnectionSettings,
    CameraRecipeSettings,
    DeviceAvailability,
    SaveSettings,
    TriggerMode,
    TriggerSettings,
)
from devices.ccd_settings_store import CcdMachineSettingsStore
from devices.factory import CcdDevices
from devices.sapera_api import SaperaVersions
from devices.sapera_camera import ApplyNote
from devices.sapera_diagnose import DiagnoseReport, DiagnoseStep, STEP_TITLES
from devices.simulated import SimulatedLineScanCamera, SimulatedMeterWheel
from gui.ccd_controller import CcdController
from gui.sapera_diagnostics import (
    NOT_COLLECTED_STATEMENT,
    SaperaDiagnosticsReport,
    report_text,
    write_diagnostics_report,
)
from gui.sapera_location_dialog import (
    SaperaLocationCatalog,
    SaperaLocationDialog,
)
from gui.screens.ccd_screen import ACCESS_ADMIN, ACCESS_ENGINEER, CcdScreen

SERVER_A = "Xtium-CL_MX4_1"
SERVER_B = "System"


class FakeRuntime:
    """Stands in for `SaperaRuntime` without pythonnet; only versions／check_api are used."""

    def __init__(self, versions: SaperaVersions, missing: tuple[str, ...] = (), interop=None):
        self.versions = versions
        self._missing = tuple(missing)
        self._interop = interop

    def check_api(self) -> tuple[str, ...]:
        return self._missing

    def interop(self):
        if self._interop is None:
            raise AssertionError("this fake runtime has no interop")
        return self._interop


class FakeInterop:
    """Records the enumeration calls so the test can prove they went to the machine's Sapera."""

    def __init__(self, servers: dict[str, dict[str, tuple[str, ...]]]):
        self.servers = servers
        self.calls: list[tuple] = []

    def server_count(self) -> int:
        self.calls.append(("server_count",))
        return len(self.servers)

    def server_name(self, index: int) -> str:
        self.calls.append(("server_name", index))
        return list(self.servers)[index]

    def resource_count(self, server: str, kind: str) -> int:
        self.calls.append(("resource_count", server, kind))
        return len(self.servers[server][kind])

    def resource_name(self, server: str, kind: str, index: int) -> str:
        self.calls.append(("resource_name", server, kind, index))
        return self.servers[server][kind][index]


class FakeSaperaCamera(SimulatedLineScanCamera):
    """A camera that looks like the Sapera binding to `getattr(camera, "runtime", None)`."""

    def __init__(self, *, versions: SaperaVersions | None = None, availability=None, notes=(), **kwargs):
        super().__init__(**kwargs)
        self._fake_versions = versions
        self._fake_availability = availability
        self._fake_notes = tuple(notes)
        self._runtime: FakeRuntime | None = None
    def availability(self) -> DeviceAvailability:
        if self._fake_availability is not None and not self._fake_availability.available:
            return self._fake_availability
        if self._fake_versions is not None and self._runtime is None:
            self._runtime = FakeRuntime(self._fake_versions)
        if self._fake_availability is not None:
            return self._fake_availability
        return DeviceAvailability(True)

    @property
    def runtime(self):
        # Lazy, exactly like `SaperaLineScanCamera.runtime` after its first `availability()`.
        if self._runtime is None and self._fake_versions is not None:
            self._runtime = FakeRuntime(self._fake_versions)
        return self._runtime

    def apply_notes(self):
        return self._fake_notes


class _ReloadableWheel(SimulatedMeterWheel):
    """A meter wheel whose DLL arrives only after the operator points at it."""

    def __init__(self) -> None:
        super().__init__(present_card_ids=(0,))
        self.reloads = 0
        self.loaded = False

    def availability(self) -> DeviceAvailability:
        if self.loaded:
            return DeviceAvailability(True)
        return DeviceAvailability(False, "找不到 LSI-8181 DLL：LSI8181_64.dll")

    def reload_library(self) -> DeviceAvailability:
        self.reloads += 1
        self.loaded = True
        return self.availability()


def _ok_catalog(**overrides) -> SaperaLocationCatalog:
    values = dict(
        servers=(SERVER_A, SERVER_B),
        acq_resources={SERVER_A: ("Xtium-CL_MX4_1_1",), SERVER_B: ()},
        acq_devices={SERVER_A: ("Camera Link 0",), SERVER_B: ()},
        ccf_files=("C:/Sapera/CamFiles/User/line.ccf",),
        ccf_dir="C:/Sapera/CamFiles/User",
    )
    values.update(overrides)
    return SaperaLocationCatalog(**values)


def _step(code: str, status: str, short: str, details=()) -> DiagnoseStep:
    return DiagnoseStep(code, STEP_TITLES[code], status, short, tuple(details))


def _report(steps: tuple[DiagnoseStep, ...]) -> DiagnoseReport:
    counts = {status: sum(1 for step in steps if step.status == status) for status in ("PASS", "FAIL", "SKIP")}
    summary = f"S1-S8：{counts['PASS']} PASS、{counts['FAIL']} FAIL、{counts['SKIP']} SKIP"
    return DiagnoseReport(steps, "outputs/logs/camera/r.txt", "outputs/logs/camera/r.json", summary, ())


def _wait_until(app: QApplication, predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return predicate()


class SaperaGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.store = CcdMachineSettingsStore(self.root / "config" / "ccd_machine.json")
        self.log_dir = self.root / "camera_logs"
        self.meter_wheel = SimulatedMeterWheel(present_card_ids=(0,))

    def tearDown(self):
        self._temp.cleanup()

    # ------------------------------------------------------------------
    def _screen(self, camera) -> tuple[CcdScreen, CcdController]:
        screen = CcdScreen()
        controller = CcdController(CcdDevices(camera, self.meter_wheel), self.store)
        controller.diagnostics_log_dir = self.log_dir
        controller.attach(screen)
        self.addCleanup(controller.close)
        return screen, controller

    def _dialog(self, catalog: SaperaLocationCatalog, current: CameraConnectionSettings | None = None):
        probe_calls = []

        def probe():
            probe_calls.append(1)
            return catalog

        dialog = SaperaLocationDialog(probe, current=current)
        self.addCleanup(dialog.deleteLater)
        return dialog, probe_calls

    # ------------------------------------------------------------------
    # A. location dialog
    # ------------------------------------------------------------------
    def test_location_dialog_probes_through_an_injected_catalog(self):
        dialog, probe_calls = self._dialog(_ok_catalog())
        self.assertEqual(probe_calls, [1], "the injected prober is what fills the dialog")
        # Only Acq-capable servers are offered: `System` is Sapera's host pseudo-server and has no
        # Acq resource, so offering it is what produced the camera machine's E-0502 report.
        self.assertEqual([dialog.server_combo.itemText(i) for i in range(dialog.server_combo.count())],
                         [SERVER_A])
        self.assertEqual(dialog.selected_server(), SERVER_A)
        self.assertIn("已略過沒有 Acq resource", dialog.reason_label.text())
        self.assertIn(SERVER_B, dialog.reason_label.text())
        self.assertEqual(dialog.acq_combo.count(), 1)
        self.assertEqual(
            [dialog.ccf_list.item(i).text() for i in range(dialog.ccf_list.count())],
            ["C:/Sapera/CamFiles/User/line.ccf"],
        )

    def test_location_dialog_never_defaults_to_a_host_server_without_acq_resources(self):
        """Field report: `System` was preselected, so SapAcquisition could never be created."""

        catalog = _ok_catalog(servers=(SERVER_B, SERVER_A))  # `System` enumerated first, as on site
        dialog, _calls = self._dialog(catalog)
        self.assertEqual(dialog.selected_server(), SERVER_A)
        self.assertEqual(dialog.selected_resource_index(), 0)
        self.assertEqual(dialog.server_combo.count(), 1)

    def test_location_dialog_prefers_the_saved_capture_server(self):
        catalog = _ok_catalog(
            servers=(SERVER_A, "Xtium-CL_MX4_2"),
            acq_resources={SERVER_A: ("a",), "Xtium-CL_MX4_2": ("b",)},
        )
        dialog, _calls = self._dialog(
            catalog, current=CameraConnectionSettings(server_name="Xtium-CL_MX4_2", resource_index=0)
        )
        self.assertEqual(dialog.selected_server(), "Xtium-CL_MX4_2")

    def test_location_dialog_still_lists_servers_when_none_has_an_acq_resource(self):
        dialog, _calls = self._dialog(
            _ok_catalog(servers=(SERVER_B,), acq_resources={SERVER_B: ()}, acq_devices={})
        )
        self.assertEqual(dialog.server_combo.count(), 1, "the operator must see what Sapera reported")
        self.assertEqual(dialog.selected_server(), SERVER_B)

    def test_location_dialog_auto_selects_a_single_acq_device_with_a_hint(self):
        dialog, _calls = self._dialog(_ok_catalog())
        self.assertEqual(len(dialog.device_group), 1)
        self.assertTrue(dialog.device_group[0].isChecked())
        self.assertIn("已自動選取", dialog.device_hint.text())
        self.assertIn(SERVER_A, dialog.device_hint.text())
        self.assertEqual(dialog.selected_device(), (SERVER_A, 0))
        self.assertTrue(dialog.save_button.isEnabled())

    def test_location_dialog_requires_an_explicit_choice_for_two_acq_devices(self):
        catalog = _ok_catalog(
            acq_devices={SERVER_A: ("Camera Link 0", "Camera Link 1"), SERVER_B: ("Virtual 0",)},
        )
        dialog, _calls = self._dialog(catalog)
        self.assertEqual(len(dialog.device_group), 3)
        self.assertFalse(any(button.isChecked() for button in dialog.device_group), "no guessing")
        self.assertIn("3 個 AcqDevice", dialog.device_hint.text())
        for button in dialog.device_group:
            self.assertIn("resource", button.text())
        # Both device servers are listed, not only the first one.
        joined = " ".join(button.text() for button in dialog.device_group)
        self.assertIn(SERVER_A, joined)
        self.assertIn(SERVER_B, joined)
        dialog.device_group[2].setChecked(True)
        self.assertEqual(dialog.selected_device(), (SERVER_B, 0))

    def test_location_dialog_shows_the_unavailable_reason_and_can_be_rejected(self):
        dialog, _calls = self._dialog(
            SaperaLocationCatalog(unavailable_reason="E-0104 找不到 Sapera LT 安裝目錄")
        )
        self.assertFalse(dialog.reason_label.isHidden())
        self.assertIn("E-0104", dialog.reason_label.text())
        self.assertFalse(dialog.save_button.isEnabled(), "nothing to pick without Sapera")
        dialog.reject()
        self.assertEqual(dialog.result(), 0)

    def test_location_dialog_survives_a_raising_prober(self):
        def probe():
            raise RuntimeError("boom")

        dialog = SaperaLocationDialog(probe)
        self.addCleanup(dialog.deleteLater)
        self.assertFalse(dialog.reason_label.isHidden())
        self.assertIn("E-0401", dialog.reason_label.text())
        self.assertFalse(dialog.save_button.isEnabled())

    def test_accepted_dialog_writes_into_the_machine_settings_store(self):
        camera = FakeSaperaCamera(versions=SaperaVersions(assembly_file_version="8.60.0.0"))
        screen, controller = self._screen(camera)
        screen.set_mode("admin")
        controller.sapera_location_prober = lambda: _ok_catalog()
        accepted = {"value": SaperaLocationDialog.DialogCode.Accepted}

        def fake_exec(dialog):
            dialog.server_combo.setCurrentIndex(0)
            dialog.acq_combo.setCurrentIndex(0)
            dialog.ccf_edit.setText("C:/Sapera/CamFiles/User/line.ccf")
            dialog.device_group[0].setChecked(True)
            return accepted["value"]

        with patch.object(SaperaLocationDialog, "exec", fake_exec):
            screen.choose_location_button.click()
        saved = self.store.load().connection
        self.assertEqual(saved.server_name, SERVER_A)
        self.assertEqual(saved.resource_index, 0)
        self.assertEqual(saved.config_file_path, "C:/Sapera/CamFiles/User/line.ccf")
        self.assertEqual(saved.device_feature_server_name, SERVER_A)
        self.assertEqual(saved.device_feature_resource_index, 0)
        # The existing 連線／套用 flow owns the canonical write path.
        screen.apply_camera_button.click()
        self.assertEqual(self.store.load().connection.server_name, SERVER_A)

    def test_cancelled_dialog_leaves_the_settings_untouched(self):
        camera = FakeSaperaCamera(versions=SaperaVersions(assembly_file_version="8.60.0.0"))
        screen, controller = self._screen(camera)
        screen.set_mode("admin")
        controller.sapera_location_prober = lambda: _ok_catalog()
        with patch.object(SaperaLocationDialog, "exec", lambda dialog: SaperaLocationDialog.DialogCode.Rejected):
            screen.choose_location_button.click()
        self.assertFalse(self.store.path.exists(), "cancel must not write the machine settings file")

    def test_controller_probe_enumerates_through_the_machine_interop(self):
        interop = FakeInterop({
            SERVER_A: {"Acq": ("Xtium-CL_MX4_1_1",), "AcqDevice": ("Camera Link 0",)},
            SERVER_B: {"Acq": (), "AcqDevice": ()},
        })
        versions = SaperaVersions(
            assembly_file_version="8.60.0.0",
            native_file_version="8.60.0.0",
            assembly_path=str(self.root / "Components" / "NET" / "Bin" / "SapClassBasic.dll"),
        )
        camera = FakeSaperaCamera(versions=versions)
        camera._runtime = FakeRuntime(versions, interop=interop)
        controller = CcdController(CcdDevices(camera, self.meter_wheel), self.store)
        self.addCleanup(controller.close)
        catalog = controller.probe_sapera_locations(camera)
        self.assertEqual(catalog.servers, (SERVER_A, SERVER_B))
        self.assertEqual(catalog.acq_resources[SERVER_A], ("Xtium-CL_MX4_1_1",))
        self.assertEqual(catalog.device_options(), ((SERVER_A, 0, "Camera Link 0"),))
        self.assertIn(("resource_count", SERVER_A, "Acq"), interop.calls)
        self.assertIn(("resource_name", SERVER_A, "AcqDevice", 0), interop.calls)

    def test_controller_probe_reports_missing_api_members_instead_of_raising(self):
        versions = SaperaVersions(assembly_file_version="9.12.0.0", native_file_version="8.60.0.0")
        camera = FakeSaperaCamera(versions=versions)
        camera._runtime = FakeRuntime(versions, missing=("SapManager.GetServerCount()",))
        controller = CcdController(CcdDevices(camera, self.meter_wheel), self.store)
        self.addCleanup(controller.close)
        catalog = controller.probe_sapera_locations(camera)
        self.assertFalse(catalog.available)
        self.assertIn("E-0301", catalog.unavailable_reason)
        self.assertIn("SapManager.GetServerCount()", catalog.unavailable_reason)

    def test_controller_probe_without_runtime_reports_the_reason(self):
        camera = FakeSaperaCamera(
            availability=DeviceAvailability(False, "E-0104 找不到 Sapera LT 安裝目錄")
        )
        controller = CcdController(CcdDevices(camera, self.meter_wheel), self.store)
        self.addCleanup(controller.close)
        catalog = controller.probe_sapera_locations(camera)
        self.assertFalse(catalog.available)
        self.assertIn("E-0104", catalog.unavailable_reason)

    # ------------------------------------------------------------------
    # B. version mismatch and API self-check
    # ------------------------------------------------------------------

    def test_screen_asks_the_window_for_the_meter_wheel_dll(self):
        """The camera machine cannot set environment variables, so the DLL is chosen in the GUI."""

        screen, _controller = self._screen(SimulatedLineScanCamera(auto_emit=False))
        screen.set_mode("admin")
        requested: list[int] = []
        screen.meter_wheel_dll_requested.connect(lambda: requested.append(1))
        screen.meter_wheel_dll_button.click()
        self.assertEqual(requested, [1])

    def test_controller_saves_the_meter_wheel_dll_path_and_retries_the_load(self):
        wheel = _ReloadableWheel()
        screen = CcdScreen()
        controller = CcdController(CcdDevices(SimulatedLineScanCamera(auto_emit=False), wheel), self.store)
        controller.attach(screen)
        self.addCleanup(controller.close)

        self.assertFalse(controller.availability()[1].available)
        self.assertTrue(controller.set_meter_wheel_dll_path(r"C:\vendor\LSI8181_64.dll"))

        self.assertEqual(wheel.reloads, 1, "the new path must be tried immediately")
        self.assertEqual(self.store.load().meter_wheel.dll_path, r"C:\vendor\LSI8181_64.dll")
        self.assertTrue(controller.availability()[1].available)

    def test_the_meter_wheel_dll_row_is_admin_only(self):
        screen, _controller = self._screen(SimulatedLineScanCamera(auto_emit=False))
        screen.set_mode("admin")
        self.assertTrue(screen.meter_wheel_dll_button.isEnabled())
        screen.set_mode("eng")
        self.assertFalse(screen.meter_wheel_dll_button.isEnabled())
    def test_version_mismatch_is_shown_but_the_camera_stays_available(self):
        camera = FakeSaperaCamera(
            versions=SaperaVersions(
                assembly_file_version="9.12.0.0",
                native_file_version="8.60.0.0",
                assembly_path="C:/Sapera/Components/NET/Bin/SapClassBasic.dll",
            )
        )
        screen, controller = self._screen(camera)
        availability = camera.availability()
        self.assertTrue(availability.available, "a version mismatch is never 沒有擷取卡")
        self.assertFalse(screen.sapera_version_notice.isHidden())
        notice = screen.sapera_version_notice.text()
        self.assertIn("Sapera runtime 版本不符", notice)
        self.assertIn("9.12.0.0", notice)
        self.assertIn("8.60.0.0", notice)
        self.assertIn("9.12.0.0", screen.sapera_managed_label.text())
        self.assertIn("8.60.0.0", screen.sapera_native_label.text())

    def test_unknown_versions_do_not_show_a_mismatch_notice(self):
        camera = FakeSaperaCamera(versions=SaperaVersions())
        screen, controller = self._screen(camera)
        self.assertTrue(screen.sapera_version_notice.isHidden())
        self.assertEqual(screen.sapera_managed_label.text(), "—")
        self.assertEqual(screen.sapera_native_label.text(), "—")

    def test_simulator_and_unavailable_backend_behave_exactly_as_before(self):
        screen = CcdScreen()
        controller = CcdController(
            CcdDevices(SimulatedLineScanCamera(auto_emit=False), self.meter_wheel), self.store
        )
        self.addCleanup(controller.close)
        controller.attach(screen)
        self.assertTrue(screen.sapera_version_notice.isHidden())
        self.assertTrue(screen.sapera_api_label.isHidden())
        self.assertEqual(screen.sapera_managed_label.text(), "—")
        view = controller.sapera_versions_view()
        self.assertFalse(view.known)
        self.assertFalse(view.mismatch)

    def test_window_shows_one_mismatch_notice_per_session(self):
        camera = FakeSaperaCamera(
            versions=SaperaVersions(assembly_file_version="9.12.0.0", native_file_version="8.60.0.0")
        )
        controller = CcdController(CcdDevices(camera, self.meter_wheel), self.store)
        self.addCleanup(controller.close)
        notices: list[tuple[str, str]] = []
        controller.notice.connect(lambda message, kind: notices.append((message, kind)))
        controller.refresh_sapera_versions()
        controller.refresh_sapera_versions()
        controller.refresh_sapera_versions()
        self.assertEqual(len(notices), 1, "the session notice is emitted once")
        self.assertIn("Sapera runtime 版本不符", notices[0][0])
        self.assertIn("9.12.0.0", notices[0][0])
        self.assertIn("8.60.0.0", notices[0][0])
        self.assertEqual(notices[0][1], "warning")

    def test_api_self_check_failure_shows_the_member_signature_on_the_page(self):
        signature = "SapManager.GetResourceCount(System.String, SapManager+ResourceType)"
        versions = SaperaVersions(assembly_file_version="8.60.0.0", native_file_version="8.60.0.0")
        camera = FakeSaperaCamera(versions=versions)
        camera._runtime = FakeRuntime(versions, missing=(signature,))
        camera._fake_availability = DeviceAvailability(False, f"E-0301 Sapera API 缺少必要成員：{signature}")
        screen, _controller = self._screen(camera)
        self.assertFalse(screen.sapera_api_label.isHidden())
        text = screen.sapera_api_label.text()
        self.assertIn("E-0301", text)
        self.assertIn(signature, text)
        self.assertIn("8.60.0.0", screen.sapera_managed_label.text())
        self.assertIn("相機不可用", screen.camera_availability_label.text())

    # ------------------------------------------------------------------
    # C. GUI diagnose run
    # ------------------------------------------------------------------
    def _diagnose_steps(self, fail: bool = False) -> tuple[DiagnoseStep, ...]:
        steps = [
            _step("S1", "PASS", "S1 PASS Sapera 8.60.0.0"),
            _step("S2", "PASS", "S2 PASS managed 8.60.0.0／runtime 8.60.0.0"),
            _step("S3", "PASS", "S3 PASS API 成員齊全"),
            _step("S4", "PASS", "S4 PASS 2 個 server、CCF 1 個"),
            _step("S5", "PASS", "S5 PASS 建立並釋放 4 個物件"),
            _step("S6", "PASS", "S6 PASS 參數寫入並讀回 6 項"),
            _step("S7", "PASS", "S7 PASS 32×720 min0 max255 mean12.0"),
            _step("S8", "PASS", "S8 PASS 已斷線並清理"),
        ]
        if fail:
            steps[5] = _step("S6", "FAIL", "S6 FAIL E-0602 Exposure 寫入失敗")
            steps[6] = _step("S7", "SKIP", "S7 SKIP 前一步失敗")
            steps[7] = _step("S8", "SKIP", "S8 SKIP S6 未連線")
        return tuple(steps)

    def test_diagnose_run_shows_every_short_line_from_a_worker_thread(self):
        camera = FakeSaperaCamera(versions=SaperaVersions(assembly_file_version="8.60.0.0"))
        screen, controller = self._screen(camera)
        screen.set_mode("admin")
        calls: list[tuple] = []
        runner_threads: list[int] = []
        report = _report(self._diagnose_steps())

        def runner(**kwargs):
            calls.append(tuple(sorted(kwargs)))
            runner_threads.append(threading.get_ident())
            return report

        controller.diagnose_runner = runner
        main_thread = threading.get_ident()
        self.assertTrue(controller.start_camera_diagnose())
        self.assertFalse(screen.diagnose_button.isEnabled(), "disabled while a run is in progress")
        self.assertTrue(_wait_until(self.app, lambda: not controller.diagnose_running))
        self.assertTrue(_wait_until(self.app, lambda: not screen.sapera_diagnose_result_label.isHidden()))
        self.assertTrue(_wait_until(self.app, lambda: screen.diagnose_button.isEnabled()))
        text = screen.sapera_diagnose_result_label.text()
        for line in report.lines():
            self.assertIn(line, text)
        self.assertIn(report.summary(), text)
        self.assertEqual(runner_threads and len(runner_threads), 1)
        self.assertNotEqual(runner_threads[0], main_thread, "S1-S8 must not run on the GUI thread")
        self.assertEqual(calls[0][0], "acquisition")

    def test_a_second_concurrent_diagnose_start_is_refused(self):
        camera = FakeSaperaCamera(versions=SaperaVersions(assembly_file_version="8.60.0.0"))
        screen, controller = self._screen(camera)
        screen.set_mode("admin")
        blocker = threading.Event()
        started = threading.Event()

        def runner(**_kwargs):
            started.set()
            blocker.wait(5)
            return _report(self._diagnose_steps())

        controller.diagnose_runner = runner
        notices: list[tuple[str, str]] = []
        controller.notice.connect(lambda message, kind: notices.append((message, kind)))
        self.assertTrue(controller.start_camera_diagnose())
        self.assertTrue(_wait_until(self.app, started.is_set))
        self.assertFalse(controller.start_camera_diagnose(), "the second run must be refused")
        self.assertTrue(any("正在執行中" in message for message, _kind in notices))
        self.assertFalse(screen.diagnose_button.isEnabled())
        blocker.set()
        self.assertTrue(_wait_until(self.app, lambda: not controller.diagnose_running))

    def test_failing_report_keeps_the_failing_short_code_visible(self):
        camera = FakeSaperaCamera(versions=SaperaVersions(assembly_file_version="8.60.0.0"))
        screen, controller = self._screen(camera)
        screen.set_mode("admin")
        controller.diagnose_runner = lambda **_kwargs: _report(self._diagnose_steps(fail=True))
        notices: list[tuple[str, str]] = []
        controller.notice.connect(lambda message, kind: notices.append((message, kind)))
        self.assertTrue(controller.start_camera_diagnose())
        self.assertTrue(_wait_until(self.app, lambda: not controller.diagnose_running))
        self.assertTrue(_wait_until(self.app, lambda: not screen.sapera_diagnose_result_label.isHidden()))
        text = screen.sapera_diagnose_result_label.text()
        self.assertIn("S6 FAIL E-0602 Exposure 寫入失敗", text)
        self.assertIn("S7 SKIP", text)
        self.assertIn("FAIL 步驟：S6", text)
        self.assertTrue(any(kind == "error" for _message, kind in notices), "a failed run is an error notice")

    def test_closing_the_window_with_a_run_in_flight_does_not_crash(self):
        camera = FakeSaperaCamera(versions=SaperaVersions(assembly_file_version="8.60.0.0"))
        screen, controller = self._screen(camera)
        screen.set_mode("admin")
        started = threading.Event()

        def runner(**_kwargs):
            started.set()
            time.sleep(0.2)
            return _report(self._diagnose_steps())

        controller.diagnose_runner = runner
        self.assertTrue(controller.start_camera_diagnose())
        self.assertTrue(_wait_until(self.app, started.is_set))
        controller.close()
        self.assertFalse(controller.diagnose_running)
        self.assertIsNone(controller._diagnose_controller.thread)
        self.assertTrue(screen.diagnose_button is not None)

    # ------------------------------------------------------------------
    # D. 匯出診斷
    # ------------------------------------------------------------------
    def test_export_writes_one_utf8_report_with_the_collected_facts(self):
        notes = (
            ApplyNote("Exposure", "ExposureTime=1200 讀回 1200"),
            ApplyNote("Length", "CROP_HEIGHT=720 寫入失敗", "E-0604"),
        )
        camera = FakeSaperaCamera(
            versions=SaperaVersions(
                assembly_file_version="9.12.0.0",
                native_file_version="8.60.0.0",
                assembly_path="C:/Sapera/Components/NET/Bin/SapClassBasic.dll",
            ),
            notes=notes,
        )
        screen, controller = self._screen(camera)
        screen.set_mode("admin")
        controller.apply_camera_settings(
            CameraConnectionSettings(SERVER_A, 1, "C:/Sapera/CamFiles/User/line.ccf"),
            CameraRecipeSettings(
                AcquisitionSettings(2200, 3.5, 4096, 120),
                TriggerSettings(TriggerMode.EXTERNAL, True, False, False),
            ),
        )
        controller.diagnose_runner = lambda **_kwargs: _report(self._diagnose_steps(fail=True))
        self.assertTrue(controller.start_camera_diagnose())
        self.assertTrue(_wait_until(self.app, lambda: not controller.diagnose_running))
        self.assertTrue(_wait_until(self.app, lambda: not screen.sapera_diagnose_result_label.isHidden()))

        notices: list[tuple[str, str]] = []
        controller.notice.connect(lambda message, kind: notices.append((message, kind)))
        path = controller.export_camera_diagnostics()
        self.assertIsNotNone(path)
        self.assertEqual(Path(path).parent, self.log_dir)
        self.assertTrue(Path(path).is_file())
        raw = Path(path).read_bytes()
        text = raw.decode("utf-8")
        self.assertIn(SERVER_A, text)
        self.assertIn("C:/Sapera/CamFiles/User/line.ccf", text)
        self.assertIn("2200", text)
        self.assertIn(TRIGGER_MODE_LABELS[TriggerMode.EXTERNAL], text)
        for note in notes:
            self.assertIn(note.line(), text)
        self.assertIn("9.12.0.0", text)
        self.assertIn("8.60.0.0", text)
        self.assertIn("Sapera runtime 版本不符", text)
        self.assertIn("S6 FAIL E-0602 Exposure 寫入失敗", text)
        self.assertIn("未收集", text)
        self.assertIn("Live Features", text)
        self.assertIn("Acq Params", text)
        self.assertTrue(any(str(path) in _message for _message, _kind in notices))
        # Nothing is written outside the injected directory.
        self.assertEqual(sorted(p.name for p in self.root.rglob("*.txt")), sorted(p.name for p in self.log_dir.glob("*.txt")))

    def test_export_is_refused_without_a_sapera_backend(self):
        screen, controller = self._screen(SimulatedLineScanCamera(auto_emit=False))
        screen.set_mode("admin")
        notices: list[tuple[str, str]] = []
        controller.notice.connect(lambda message, kind: notices.append((message, kind)))
        self.assertIsNone(controller.export_camera_diagnostics())
        self.assertEqual(len(notices), 1)
        self.assertIn("Sapera", notices[0][0])
        self.assertFalse(self.log_dir.exists())

    def test_report_text_names_the_uncollected_items(self):
        report = SaperaDiagnosticsReport(
            connection=CameraConnectionSettings(),
            product=CameraRecipeSettings(),
            apply_notes=(),
            versions=SaperaVersions(),
            versions_summary="managed 未知／runtime 未知",
            versions_mismatch=False,
            availability_reason="",
        )
        text = report_text(report)
        self.assertIn("== 未收集 ==", text)
        # The statement is indented inside its section; compare the stripped line.
        self.assertIn(NOT_COLLECTED_STATEMENT, [line.strip() for line in text.splitlines()])

    def test_write_diagnostics_report_uses_the_diagnose_log_subdir_by_default(self):
        from devices.sapera_diagnose import DIAGNOSE_LOG_SUBDIR

        report = SaperaDiagnosticsReport(
            connection=CameraConnectionSettings(),
            product=CameraRecipeSettings(),
            apply_notes=(),
            versions=SaperaVersions(),
            versions_summary="",
            versions_mismatch=False,
            availability_reason="",
        )
        target = self.root / "subdir-check"
        path = write_diagnostics_report(report, log_dir=target)
        self.assertEqual(path.parent, target)
        self.assertTrue(path.is_file())
        self.assertTrue(str(DIAGNOSE_LOG_SUBDIR).endswith(str(Path("outputs") / "logs" / "camera")))

    # ------------------------------------------------------------------
    # E. permissions and keyboard
    # ------------------------------------------------------------------
    def test_new_controls_are_admin_only_and_hidden_from_op(self):
        camera = FakeSaperaCamera(versions=SaperaVersions(assembly_file_version="8.60.0.0"))
        screen, _controller = self._screen(camera)
        screen.set_mode("eng")
        self.assertFalse(screen.diagnose_button.isEnabled())
        self.assertFalse(screen.export_diagnostics_button.isEnabled())
        self.assertFalse(screen.choose_location_button.isEnabled())
        self.assertEqual(screen.gate.access_of(screen.diagnose_button), ACCESS_ADMIN)
        self.assertEqual(screen.gate.access_of(screen.choose_location_button), ACCESS_ADMIN)
        self.assertTrue(screen.diagnostics_panel.isHidden())
        # Engineer keeps everything they have today.
        self.assertTrue(screen.connect_button.isEnabled())

        screen.set_mode("admin")
        self.assertTrue(screen.diagnose_button.isEnabled())
        self.assertFalse(screen.diagnostics_panel.isHidden())

        screen.set_mode("op")
        self.assertTrue(screen.diagnostics_panel.isHidden())
        self.assertFalse(screen.diagnose_button.isEnabled())

    def test_op_cannot_reach_the_location_dialog_through_the_screen(self):
        camera = FakeSaperaCamera(versions=SaperaVersions(assembly_file_version="8.60.0.0"))
        screen, controller = self._screen(camera)
        screen.set_mode("op")
        emitted: list[object] = []
        screen.sapera_location_requested.connect(emitted.append)
        screen.choose_location_button.click()
        self.assertEqual(emitted, [], "a disabled button must not request the dialog")

    def test_keyboard_space_activates_the_new_admin_controls(self):
        camera = FakeSaperaCamera(versions=SaperaVersions(assembly_file_version="8.60.0.0"))
        screen, controller = self._screen(camera)
        screen.set_mode("admin")
        controller.diagnose_runner = lambda **_kwargs: _report(self._diagnose_steps())
        for button in (screen.diagnose_button, screen.export_diagnostics_button, screen.choose_location_button):
            self.assertNotEqual(button.focusPolicy(), Qt.FocusPolicy.NoFocus, button.text())
        screen.diagnose_button.setFocus()
        QTest.keyClick(screen.diagnose_button, Qt.Key.Key_Space)
        self.assertTrue(controller.diagnose_running or controller._last_diagnose_report is not None)
        self.assertTrue(_wait_until(self.app, lambda: not controller.diagnose_running))
        self.assertTrue(_wait_until(self.app, lambda: screen.diagnose_button.isEnabled()))

        exported: list[Path | None] = []
        screen.sapera_diagnostics_export_requested.connect(
            lambda: exported.append(controller.export_camera_diagnostics())
        )
        screen.export_diagnostics_button.setFocus()
        QTest.keyClick(screen.export_diagnostics_button, Qt.Key.Key_Space)
        self.assertEqual(len(exported), 1)
        self.assertTrue(exported[0] is not None and Path(exported[0]).is_file())
        self.assertEqual(Path(exported[0]).parent, self.log_dir)

    def test_version_labels_keep_the_backend_neutral_interface(self):
        """`devices/interfaces.py` must not gain a Sapera-only member."""

        from devices.interfaces import LineScanCamera

        names = {name for name in dir(LineScanCamera)}
        self.assertNotIn("runtime", names)
        self.assertNotIn("apply_notes", names)


if __name__ == "__main__":
    unittest.main()
