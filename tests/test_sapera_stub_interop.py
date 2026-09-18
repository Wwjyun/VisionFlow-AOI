"""Real-pythonnet integration test for the Sapera binding, using a compiled stand-in assembly.

`tests/fixtures/sapera_stub/SapClassBasicStub.cs` reproduces the public shapes that
`devices/sapera_api.py` calls (overloads, out parameters, nested enums, thread-raised events,
`IntPtr` ReadRect). It is compiled here with the .NET Framework `csc.exe` and loaded through the
production `load_runtime()` path, so this module exercises pythonnet itself: assembly load, API
manifest reflection, overload selection, `SapAcquisition`/`SapBufferWithTrash`/`SapAcqToBuf`
lifecycle, end-of-frame handoff from a non-UI thread, and the external-trigger event mapping.

It never needs Sapera LT, a frame grabber, or a camera: the whole module is skipped when `csc.exe`
or pythonnet is unavailable, and it proves nothing about real Sapera timing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUB_SOURCE = ROOT / "tests" / "fixtures" / "sapera_stub" / "SapClassBasicStub.cs"
SERVER = "Xtium-CL_MX4_1"
# The stand-in reports these two bounds as present but zero; real hardware reports real limits.
CAM_LINE_RATE_BOUNDS = ("CAM_LINE_TRIGGER_FREQ_MIN", "CAM_LINE_TRIGGER_FREQ_MAX")

_STUB: dict[str, object] = {}


def _find_csc() -> Path | None:
    override = os.environ.get("VISIONFLOW_CSC")
    candidates = [Path(override)] if override else []
    windir = os.environ.get("SystemRoot") or r"C:\Windows"
    for framework in ("Framework64", "Framework"):
        candidates.append(Path(windir) / "Microsoft.NET" / framework / "v4.0.30319" / "csc.exe")
    found = shutil.which("csc")
    if found:
        candidates.append(Path(found))
    return next((path for path in candidates if path.is_file()), None)


def setUpModule() -> None:
    if not STUB_SOURCE.is_file():
        _STUB["error"] = f"缺少測試用 stub 原始碼：{STUB_SOURCE}"
        return
    csc = _find_csc()
    if csc is None:
        _STUB["error"] = "找不到 .NET Framework csc.exe（可設定 VISIONFLOW_CSC）"
        return
    try:
        import pythonnet  # noqa: F401, PLC0415
        import clr  # noqa: F401, PLC0415
    except ImportError as exc:
        _STUB["error"] = f"未安裝 pythonnet：{exc}"
        return

    # The loaded assembly stays locked for the life of the process, so its directory is created
    # outside the test and its cleanup failure is ignored on purpose.
    directory = tempfile.TemporaryDirectory(prefix="visionflow-sapera-stub-", ignore_cleanup_errors=True)
    _STUB["directory"] = directory
    dll = Path(directory.name) / "DALSA.SaperaLT.SapClassBasic.dll"
    try:
        result = subprocess.run(
            [str(csc), "/nologo", "/target:library", f"/out:{dll}", str(STUB_SOURCE)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            _STUB["error"] = f"csc 編譯失敗：{(result.stdout + result.stderr).strip()[:400]}"
            return
        from devices.sapera_api import load_runtime  # noqa: PLC0415

        _STUB["runtime"] = load_runtime(dll_path=dll)
    except Exception as exc:  # noqa: BLE001 - any failure skips this module instead of failing it
        _STUB["error"] = f"{type(exc).__name__}: {exc}"


def tearDownModule() -> None:
    directory = _STUB.pop("directory", None)
    if directory is not None:
        directory.cleanup()
    _STUB.clear()


class SaperaStubInteropTests(unittest.TestCase):
    """Every test drives the production interop against the compiled stand-in assembly."""

    def setUp(self) -> None:
        runtime = _STUB.get("runtime")
        if runtime is None:
            self.skipTest(str(_STUB.get("error") or "Sapera stub 未就緒"))
        from devices.sapera_api import PythonnetSaperaInterop  # noqa: PLC0415
        from devices.sapera_camera import SaperaLineScanCamera  # noqa: PLC0415

        self.assertIsInstance(runtime.interop(), PythonnetSaperaInterop)
        self.runtime = runtime
        self.stub = runtime.namespace.StubHardware
        self.stub.Reset()
        self.stub.TakeCalls()
        self._workdir = tempfile.TemporaryDirectory(prefix="visionflow-sapera-ccf-")
        self.addCleanup(self._workdir.cleanup)
        self.ccf = Path(self._workdir.name) / "test.ccf"
        self.ccf.write_text("", encoding="utf-8")
        self.camera = SaperaLineScanCamera(interop=runtime.interop())
        self.addCleanup(self.camera.close)
        self.main_thread = threading.get_ident()

    # ---- helpers ---------------------------------------------------------------------------
    def calls(self) -> list[str]:
        return [str(call) for call in self.stub.TakeCalls()]

    def connect(self, acquisition=None, trigger=None):
        from devices.ccd_models import AcquisitionSettings, CameraConnectionSettings, TriggerSettings

        connection = CameraConnectionSettings(
            server_name=SERVER,
            resource_index=0,
            config_file_path=str(self.ccf),
        )
        return self.camera.connect(
            connection,
            acquisition or AcquisitionSettings(),
            trigger or TriggerSettings(),
        )

    def wait_for_frame(self, timeout: float = 5.0) -> np.ndarray | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frame = self.camera.latest_frame()
            if frame is not None:
                return frame
            time.sleep(0.01)
        return None

    def note_for(self, item: str) -> str:
        return next((note.line() for note in self.camera.apply_notes() if note.item == item), "")

    # ---- reflection and enumeration --------------------------------------------------------
    def test_api_manifest_is_satisfied_by_the_stand_in_assembly(self):
        self.assertEqual(self.runtime.check_api(), ())

    def test_enumeration_lists_servers_and_resources(self):
        interop = self.runtime.interop()
        self.assertEqual(interop.server_count(), 2)
        self.assertEqual(interop.server_name(0), "System")
        self.assertEqual(interop.server_name(1), SERVER)
        self.assertEqual(interop.resource_count(SERVER, "Acq"), 1)
        self.assertEqual(interop.resource_count(SERVER, "AcqDevice"), 1)
        self.assertEqual(interop.resource_count("System", "Acq"), 0)
        self.assertEqual(interop.resource_name(SERVER, "Acq", 0), "CameraLink Mono #1")

    # ---- connect write paths ---------------------------------------------------------------
    def test_continuous_connect_writes_line_rate_before_create_and_reads_back(self):
        self.stub.MissingParameters = CAM_LINE_RATE_BOUNDS
        status = self.connect()
        calls = self.calls()

        self.assertEqual(status.frame_width, 64)
        self.assertEqual(status.frame_height, 16)
        line_rate = next(i for i, call in enumerate(calls) if call.startswith("SetFeatureValue(Int64) AcquisitionLineRate="))
        create = next(i for i, call in enumerate(calls) if call.startswith("Acquisition.Create"))
        exposure = next(i for i, call in enumerate(calls) if call.startswith("SetFeatureValue(String) ExposureTime="))
        gain = next(i for i, call in enumerate(calls) if call.startswith("SetFeatureValue(String) Gain="))
        # xx_ccd order: Internal Line Rate on the camera before SapAcquisition.Create(), then Exposure/Gain.
        self.assertLess(line_rate, create)
        self.assertLess(line_rate, exposure)
        self.assertLess(exposure, gain)
        self.assertIn("SetFeatureValue(Int64) AcquisitionLineRate=30", calls)
        self.assertIn("SetFeatureValue(String) ExposureTime=1200", calls)
        self.assertIn("SetFeatureValue(String) Gain=1", calls)
        self.assertIn("UpdateFeaturesToDevice", calls)
        self.assertIn("SetParameter CROP_HEIGHT=720", calls)
        self.assertIn("SetParameter EXT_FRAME_TRIGGER_ENABLE=0", calls)
        self.assertIn("SapBufferWithTrash(2,ScatterGather)", calls)
        self.assertIn("Transfer.Create EventType=EndOfFrame", calls)
        # Board frequency is clamped to the range the card reports.
        self.assertIn("SetParameter INT_LINE_TRIGGER_ENABLE=1", calls)
        self.assertIn("SetParameter INT_LINE_TRIGGER_FREQ=100", calls)

    def test_continuous_connect_notes_record_the_readback_values(self):
        self.stub.MissingParameters = CAM_LINE_RATE_BOUNDS
        self.connect()
        self.assertIn("AcquisitionLineRate=30", self.note_for("Internal Line Rate"))
        self.assertIn("讀回 1200", self.note_for("Exposure"))
        self.assertIn("讀回 1", self.note_for("Gain"))
        self.assertIn("CROP_HEIGHT=720", self.note_for("Length"))
        self.assertIn("64×16", self.note_for("影像格式"))
        self.assertFalse([note for note in self.camera.apply_notes() if note.code])

    def test_external_trigger_connect_writes_the_confirmed_board_mapping(self):
        from devices.ccd_models import TriggerMode, TriggerSettings

        self.connect(trigger=TriggerSettings(mode=TriggerMode.EXTERNAL))
        calls = self.calls()
        self.assertIn("SetParameter LINE_INTEGRATE_METHOD=LINE_INTEGRATE_METHOD_3", calls)
        self.assertIn("SetParameter LINE_INTEGRATE_DURATION=40", calls)
        self.assertIn("SetParameter LINE_INTEGRATE_PULSE0_POLARITY=ACTIVE_HIGH", calls)
        self.assertIn("SetParameter LINE_INTEGRATE_PULSE1_POLARITY=ACTIVE_LOW", calls)
        self.assertIn("SetParameter LINE_INTEGRATE_ENABLE=1", calls)
        self.assertIn("SetParameter EXT_FRAME_TRIGGER_ENABLE=0", calls)
        # SIGNAL_NAME_PULSE1 = 0x20000 on CamIoControl[0]; the line trigger is armed after the mapping.
        self.assertIn("CamIoControl[0]=131072", calls)
        self.assertIn("SetParameter EXT_LINE_TRIGGER_ENABLE=1", calls)
        armed = calls.index("SetParameter EXT_LINE_TRIGGER_ENABLE=1")
        self.assertLess(calls.index("SetParameter LINE_INTEGRATE_ENABLE=1"), armed)
        self.assertLess(calls.index("CamIoControl[0]=131072"), armed)
        self.assertLess(calls.index("SetParameter EXT_FRAME_TRIGGER_ENABLE=0"), armed)
        # Line rate is a continuous-mode parameter and must not be written here.
        self.assertFalse([call for call in calls if "AcquisitionLineRate" in call])

    def test_software_trigger_never_enables_the_external_frame_trigger(self):
        from devices.ccd_models import TriggerMode, TriggerSettings

        self.connect(trigger=TriggerSettings(mode=TriggerMode.SOFTWARE))
        calls = self.calls()
        self.assertIn("SetParameter EXT_FRAME_TRIGGER_ENABLE=0", calls)
        self.assertIn("SetParameter EXT_LINE_TRIGGER_ENABLE=1", calls)
        self.assertNotIn("SetParameter EXT_FRAME_TRIGGER_ENABLE=1", calls)

    def test_missing_length_parameter_is_reported_but_connection_continues(self):
        self.stub.MissingParameters = CAM_LINE_RATE_BOUNDS + ("CROP_HEIGHT",)
        status = self.connect()
        self.assertIn("E-0604", self.note_for("Length"))
        self.assertIn("E-0604", status.message)

    def test_several_acq_devices_require_a_manual_feature_location(self):
        self.stub.AcqDeviceCount = 2
        self.connect()
        note = self.note_for("相機 feature")
        self.assertIn("E-0501", note)
        self.assertIn("請在「Sapera 位置」選擇", note)
        self.assertEqual(self.camera.status().state.name, "IDLE")

    # ---- frames -----------------------------------------------------------------------------
    def test_frame_arrives_from_a_driver_thread_cropped_to_width(self):
        self.stub.ExtraPitch = 16
        self.connect()
        observed: list[tuple[int, np.ndarray]] = []
        self.camera.set_frame_listener(lambda frame: observed.append((threading.get_ident(), frame.copy())))

        self.camera.capture_frame()
        frame = self.wait_for_frame()

        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (16, 64))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertFalse(frame.flags.writeable)
        expected = np.array([[(row * 7 + column + 1) & 0xFF for column in range(64)] for row in range(16)], dtype=np.uint8)
        np.testing.assert_array_equal(frame, expected)
        self.assertFalse(bool((frame == 0xEE).any()), "row padding must not reach the frame")
        self.assertEqual(len(observed), 1)
        self.assertNotEqual(observed[0][0], self.main_thread)

    def test_preview_grab_and_freeze_stop_the_transfer(self):
        from devices.ccd_models import CameraState

        self.connect()
        self.camera.start_preview()
        self.assertTrue(self.wait_for_frame() is not None)
        self.camera.stop_preview()
        calls = self.calls()
        self.assertIn("Grab", calls)
        self.assertIn("Freeze", calls)
        self.assertEqual(self.camera.status().state, CameraState.IDLE)

    def test_stop_during_capture_never_freezes_and_starts_the_cooldown(self):
        from devices.ccd_models import CameraState, DeviceError

        self.connect()
        self.camera.capture_frame()
        self.camera.stop_preview()  # capture is still in flight
        self.assertNotIn("Freeze", self.calls())
        self.assertIsNotNone(self.wait_for_frame())
        self.assertEqual(self.camera.status().state, CameraState.IDLE)
        with self.assertRaisesRegex(DeviceError, "剛停止取像"):
            self.camera.capture_frame()

    def test_trash_frame_is_reported_without_delivering_pixels(self):
        self.connect()
        observed: list[np.ndarray] = []
        self.camera.set_frame_listener(observed.append)
        # The stand-in can only tell the interop a frame is trash; drive that path directly.
        self.camera._on_frame(True)
        self.assertEqual(observed, [])
        self.assertIn("trash", self.camera.status().message)
        self.assertIsNone(self.camera.latest_frame())

    # ---- events -----------------------------------------------------------------------------
    def test_external_trigger_event_is_mapped_and_raised_from_a_driver_thread(self):
        from devices.ccd_models import TriggerMode, TriggerSettings

        self.connect(trigger=TriggerSettings(mode=TriggerMode.EXTERNAL))
        seen: list[int] = []
        delivered = threading.Event()

        def listener() -> None:
            seen.append(threading.get_ident())
            delivered.set()

        self.camera.set_external_trigger_listener(listener)
        self.stub.RaiseExternalTrigger()
        self.assertTrue(delivered.wait(5.0), "外部觸發事件未送到 listener")
        self.assertEqual(len(seen), 1)
        self.assertNotEqual(seen[0], self.main_thread)
        self.assertIn("外部觸發", self.camera.status().message)

    def test_trigger_timing_events_only_update_the_message(self):
        from devices.ccd_models import TriggerMode, TriggerSettings

        self.connect(trigger=TriggerSettings(mode=TriggerMode.EXTERNAL))
        called: list[int] = []
        self.camera.set_external_trigger_listener(lambda: called.append(1))
        self.camera._on_acq_event("ExternalTriggerTooSlow")
        self.assertEqual(called, [])
        self.assertIn("ExternalTriggerTooSlow", self.camera.status().message)

    # ---- failure and teardown ---------------------------------------------------------------
    def test_failed_acquisition_create_cleans_up_and_allows_a_retry(self):
        from devices.sapera_api import SaperaError

        self.stub.FailAcquisitionCreate = True
        with self.assertRaises(SaperaError) as caught:
            self.connect()
        self.assertEqual(caught.exception.code, "E-0502")
        calls = self.calls()
        # The temporary feature device was initialised, so it is destroyed and disposed; the failed
        # acquisition never became Initialized, so it is only disposed.
        self.assertIn("AcqDevice.Destroy", calls)
        self.assertIn("AcqDevice.Dispose", calls)
        self.assertIn("Acquisition.Dispose", calls)
        self.assertNotIn("Acquisition.Destroy", calls)
        self.assertEqual(self.camera.status().state.name, "OFFLINE")

        self.stub.FailAcquisitionCreate = False
        status = self.connect()
        self.assertEqual((status.frame_width, status.frame_height), (64, 16))

    def test_pixel_depth_other_than_eight_bits_is_rejected(self):
        from devices.sapera_api import SaperaError

        self.stub.PixelDepth = 16
        with self.assertRaises(SaperaError) as caught:
            self.connect()
        self.assertEqual(caught.exception.code, "E-0704")
        self.assertIn("PIXEL_DEPTH=16", caught.exception.detail)
        self.assertEqual(self.camera.status().state.name, "OFFLINE")

    def test_no_signal_fails_connect_and_cleans_up(self):
        from devices.sapera_api import SaperaError

        self.stub.SignalPresent = False
        with self.assertRaises(SaperaError) as caught:
            self.connect()
        self.assertEqual(caught.exception.code, "E-0505")
        self.assertIn("Acquisition.Destroy", self.calls())

    def test_pitch_without_extra_padding_still_yields_a_write_protected_frame(self):
        self.connect()
        self.camera.capture_frame()
        frame = self.wait_for_frame()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (16, 64))
        self.assertFalse(frame.flags.writeable)

    def test_disconnect_destroys_then_disposes_every_sapera_object(self):
        self.connect()
        # The temporary feature device is released during connect, not by disconnect.
        connect_calls = self.calls()
        self.assertIn("AcqDevice.Destroy", connect_calls)
        self.assertIn("AcqDevice.Dispose", connect_calls)
        self.assertNotIn("Acquisition.Destroy", connect_calls)

        self.camera.disconnect()
        calls = self.calls()
        for call in ("Transfer.Destroy", "Buffer.Destroy", "Acquisition.Destroy"):
            self.assertIn(call, calls)
        for call in ("Transfer.Dispose", "Buffer.Dispose", "Acquisition.Dispose"):
            self.assertIn(call, calls)
        order = [call for call in calls if call.endswith(("Destroy", "Dispose"))]
        self.assertLess(order.index("Transfer.Destroy"), order.index("Transfer.Dispose"))
        self.assertLess(order.index("Acquisition.Destroy"), order.index("Acquisition.Dispose"))
        self.assertFalse(bool(self.stub.LastAcquisition.Initialized))
        self.assertEqual(self.camera.status().state.name, "OFFLINE")

    def test_scatter_gather_physical_is_used_when_scatter_gather_is_unsupported(self):
        self.stub.ScatterGatherSupported = False
        self.connect()
        self.assertIn("SapBufferWithTrash(2,ScatterGatherPhysical)", self.calls())
        self.assertIn("ScatterGatherPhysical", self.note_for("影像格式"))


if __name__ == "__main__":
    unittest.main()
