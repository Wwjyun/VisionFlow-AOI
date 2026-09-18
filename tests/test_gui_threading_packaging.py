from __future__ import annotations

import os
import unittest
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from gui_launcher import bundled_recipe_path, run_packaged_gpu_fallback_smoke_test


ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = ROOT / "packaging" / "specs"
BUILD_DIR = ROOT / "packaging" / "scripts"


class GuiThreadingPackagingContractTests(unittest.TestCase):
    def test_packaged_smoke_resolves_recipe_from_pyinstaller_bundle(self):
        with tempfile.TemporaryDirectory(prefix="visionflow_bundle_") as directory:
            with patch.object(sys, "_MEIPASS", directory, create=True):
                expected = Path(directory) / "recipes" / "PRODUCT_A_AOI_01.yaml"
                self.assertEqual(bundled_recipe_path(), expected)

    def test_gui_launcher_has_noninteractive_packaged_smoke_mode(self):
        launcher = (ROOT / "gui_launcher.py").read_text(encoding="utf-8")
        spec = (SPEC_DIR / "VisionFlow AOI.spec").read_text(encoding="utf-8")

        self.assertIn('"--smoke-test" in sys.argv[1:]', launcher)
        self.assertIn("run_packaged_yolox_smoke_test", launcher)
        self.assertIn("models/yolox", spec)
        self.assertIn("window.recipe_panel.load_recipe(recipe_path)", launcher)
        self.assertIn("window.recipe_panel.detector_list.count() > 0", launcher)

    def test_packaged_smoke_uses_isolated_gui_settings(self):
        import gui.main_window
        import gui_launcher

        created = []
        real_main_window = gui.main_window.MainWindow

        def recording_main_window(*args, **kwargs):
            window = real_main_window(*args, **kwargs)
            created.append((kwargs.get("settings"), window))
            return window

        with patch.object(gui.main_window, "MainWindow", recording_main_window), \
                patch.object(gui_launcher, "run_packaged_gpu_fallback_smoke_test", return_value=0), \
                patch.object(gui_launcher, "run_packaged_yolox_smoke_test", return_value=0):
            self.assertEqual(gui_launcher.run_packaged_smoke_test(), 0)

        settings, window = created[0]
        self.assertIsNotNone(settings, "the smoke must never read or write the operator's QSettings")
        self.assertIn("visionflow_smoke_settings_", settings.fileName())
        window._inspection_gpu_sessions.close()
        window.deleteLater()

    def test_packaged_smoke_exercises_missing_dll_fallback_policy(self):
        self.assertEqual(run_packaged_gpu_fallback_smoke_test(), 0)

    def test_cuda_pipeline_workers_are_moved_to_qthreads_before_start(self):
        window_source = (ROOT / "gui" / "main_window.py").read_text(encoding="utf-8")
        controller_source = (ROOT / "gui" / "workflow_controllers.py").read_text(encoding="utf-8")
        move = "worker.moveToThread(thread)"
        started = "thread.started.connect(worker.run)"
        self.assertIn(move, controller_source)
        self.assertIn(started, controller_source)
        self.assertLess(controller_source.index(move), controller_source.index(started))
        self.assertIn("InspectionWorkflowController", window_source)
        self.assertIn("MonitorWorkflowController", window_source)
        self.assertNotIn(".wait(", window_source + controller_source)

    def test_worker_error_progress_and_monitor_cancel_use_signals_or_callback(self):
        workers = (ROOT / "gui" / "workers.py").read_text(encoding="utf-8")

        self.assertIn("failed = Signal(str)", workers)
        self.assertIn("progress = Signal(int, str)", workers)
        self.assertIn("stop_callback=lambda: self._stop_requested", workers)
        self.assertIn("self.failed.emit(str(exc))", workers)

    def test_pyinstaller_cuda_dll_is_optional_and_keeps_gpu_relative_path(self):
        spec = (SPEC_DIR / "VisionFlow AOI.spec").read_text(encoding="utf-8")
        build = (BUILD_DIR / "build_exe.ps1").read_text(encoding="utf-8")

        self.assertIn("if cuda_dll.exists() else []", spec)
        self.assertIn("(str(cuda_dll), 'gpu')", spec)
        self.assertIn("if (Test-Path $cudaDll)", build)
        self.assertIn('"VisionFlow AOI.spec"', build)
        self.assertIn("-m PyInstaller --noconfirm --clean $spec", build)
        self.assertNotIn('"--add-binary"', build)
        self.assertIn("CPU-compatible package", build)

    def test_pyinstaller_bundles_pythonnet_but_never_the_vendor_camera_dll(self):
        spec = (SPEC_DIR / "VisionFlow AOI.spec").read_text(encoding="utf-8")
        code = "\n".join(line for line in spec.splitlines() if not line.strip().startswith("#"))

        self.assertIn("'pythonnet'", spec)
        self.assertIn("'clr_loader'", spec)
        # SapClassBasic.dll and LSI8181_64.dll are installed on the camera machine, never shipped.
        self.assertNotIn("SapClassBasic", code)
        self.assertNotIn("LSI8181", code)

    def test_packaged_sapera_diagnose_short_codes_are_readable_without_a_console(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication, QPlainTextEdit

        import gui_launcher
        from devices.sapera_diagnose import DiagnoseReport, DiagnoseStep

        report = DiagnoseReport(
            steps=(
                DiagnoseStep("S1", "找到 Sapera 安裝與版本", "PASS", "S1 PASS Sapera 8.60"),
                DiagnoseStep("S2", "載入 SapClassBasic.dll", "FAIL", "S2 FAIL E-0201 找不到 DLL"),
                DiagnoseStep("S3", "API 自檢", "SKIP", "S3 SKIP 前一步失敗"),
            ),
            report_path="outputs/logs/camera/sapera-diagnose-20260918-120000.txt",
            log_path="outputs/logs/camera/sapera-diagnose-20260918-120000.json",
        )
        text = gui_launcher.sapera_diagnose_text(report)
        self.assertIn("S1 PASS Sapera 8.60", text)
        self.assertIn("S2 FAIL E-0201", text)
        self.assertIn("S3 SKIP 前一步失敗", text)
        self.assertIn("1 PASS", text)
        self.assertIn("1 FAIL", text)
        self.assertIn("sapera-diagnose-20260918-120000.txt", text)
        self.assertFalse(report.passed)

        app = QApplication.instance() or QApplication([])
        dialog = gui_launcher.build_sapera_diagnose_dialog(report)
        view = dialog.findChild(QPlainTextEdit, "sapera_diagnose_text")
        self.assertIsNotNone(view)
        self.assertTrue(view.isReadOnly())
        self.assertIn("S2 FAIL E-0201", view.toPlainText())
        dialog.deleteLater()
        app.processEvents()

    def test_packaged_sapera_diagnose_exit_code_follows_the_report(self):
        import gui_launcher
        from devices.sapera_diagnose import DiagnoseReport, DiagnoseStep

        passing = DiagnoseReport(
            steps=(DiagnoseStep("S1", "找到 Sapera 安裝與版本", "PASS", "S1 PASS Sapera 8.60"),),
            report_path="r.txt",
            log_path="r.json",
        )
        failing = DiagnoseReport(
            steps=(DiagnoseStep("S1", "找到 Sapera 安裝與版本", "FAIL", "S1 FAIL E-0104"),),
            report_path="r.txt",
            log_path="r.json",
        )
        self.assertEqual(gui_launcher.run_packaged_sapera_diagnose(lambda: passing, show_dialog=False), 0)
        self.assertEqual(gui_launcher.run_packaged_sapera_diagnose(lambda: failing, show_dialog=False), 1)
        # A report with no steps is not a pass.
        self.assertEqual(
            gui_launcher.run_packaged_sapera_diagnose(
                lambda: DiagnoseReport(steps=(), report_path="r.txt", log_path="r.json"), show_dialog=False
            ),
            1,
        )

    def test_packaged_ccd_and_diagnose_smoke_cover_a_machine_without_sapera(self):
        import gui_launcher

        self.assertEqual(gui_launcher.run_packaged_ccd_smoke_test(), 0)
        self.assertEqual(gui_launcher.run_packaged_sapera_diagnose_smoke_test(), 0)

    def test_rtx_workflow_can_accept_production_samples_and_capture_nsight(self):
        workflow = (ROOT / ".github" / "workflows" / "rtx3090-validation.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("production_manifest:", workflow)
        self.assertIn('"--production-manifest"', workflow)
        self.assertIn("Get-Command nsys", workflow)
        self.assertIn("nsys.Source profile", workflow)
        self.assertIn("outputs_validation/**/*.nsys-rep", workflow)
        self.assertIn("onnxruntime-gpu==1.27.0", workflow)
        self.assertIn("validate_yolox_ort.py", workflow)
        self.assertIn("validate_yolox_stability.py", workflow)
        self.assertIn("--iterations 1000", workflow)
        self.assertIn("yolox_acceptance_manifest", workflow)
        self.assertIn("validate_yolox_acceptance.py", workflow)


if __name__ == "__main__":
    unittest.main()
