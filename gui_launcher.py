from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
import sys
import tempfile

from gui.main_window import run_app


def bundled_recipe_path() -> Path:
    """Return a recipe from source checkout or PyInstaller's one-dir bundle."""
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return bundle_root / "recipes" / "PRODUCT_A_AOI_01.yaml"


def run_packaged_smoke_test() -> int:
    """Exercise bundled Qt startup, recipe loading, and packaged GPU fallback policy."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from gui.main_window import MainWindow

    recipe_path = bundled_recipe_path()
    if not recipe_path.is_file():
        return 2
    app = QApplication.instance() or QApplication([])
    # Isolated settings: the operator's saved image, folders and screen must neither be restored
    # (a restored large image starts background work that blocks close) nor overwritten.
    with tempfile.TemporaryDirectory(prefix="visionflow_smoke_settings_") as settings_dir:
        settings = QSettings(str(Path(settings_dir) / "gui.ini"), QSettings.Format.IniFormat)
        window = MainWindow(settings=settings)
        window.recipe_panel.load_recipe(recipe_path)
        app.processEvents()
        valid = bool(window.windowTitle()) and window.recipe_panel.detector_list.count() > 0
        window.close()
        app.processEvents()
        settings.sync()
    if not valid:
        return 3
    ccd_status = run_packaged_ccd_smoke_test()
    if ccd_status:
        return ccd_status
    diagnose_status = run_packaged_sapera_diagnose_smoke_test()
    if diagnose_status:
        return diagnose_status
    pythonnet_status = run_packaged_pythonnet_smoke_test()
    if pythonnet_status:
        return pythonnet_status
    fallback_status = run_packaged_gpu_fallback_smoke_test()
    if fallback_status:
        return fallback_status
    return run_packaged_yolox_smoke_test()


def run_packaged_pythonnet_smoke_test() -> int:
    """The camera binding needs pythonnet bundled; a machine without .NET Framework is a prerequisite gap.

    Returns 20 when `pythonnet` is missing from the bundle or the interpreter is not 64-bit (packaging
    defects), and 0 when the .NET bootstrap either succeeds or fails only because this machine has no
    .NET Framework 4.x — installing it is a documented camera-machine prerequisite, not a package bug.
    """

    try:
        import pythonnet  # noqa: F401, PLC0415
    except ImportError:
        return 20
    from devices.sapera_api import SaperaError, ensure_dotnet_runtime

    try:
        ensure_dotnet_runtime()
    except SaperaError as exc:
        return 20 if exc.code in ("E-0101", "E-0103") else 0
    return 0


def run_packaged_sapera_diagnose_smoke_test() -> int:
    """The packaged diagnosis must stop at the first missing step, say why, and never crash.

    The environment is isolated with an explicit missing assembly path, so this is the same check on
    a development machine and on the camera machine.
    """

    from devices.sapera_api import DLL_PATH_ENV as SAPERA_DLL_PATH_ENV
    from devices.sapera_diagnose import run_sapera_diagnose

    with tempfile.TemporaryDirectory(prefix="visionflow_packaged_sapera_") as temporary:
        root = Path(temporary)
        report = run_sapera_diagnose(
            environ={SAPERA_DLL_PATH_ENV: str(root / "SapClassBasic.dll")},
            log_dir=root / "logs",
        )
        reports_written = Path(report.report_path).is_file() and Path(report.log_path).is_file()
    if report.passed:
        return 16
    if not report.steps or report.steps[0].status != "FAIL" or "E-" not in report.steps[0].short:
        return 17
    if any(step.status == "PASS" for step in report.steps):
        return 18
    if not reports_written:
        return 19
    return 0


def run_packaged_ccd_smoke_test() -> int:
    """Verify a package without Sapera LT or LSI-8181 still starts and reports CCD as unavailable.

    The environment is isolated with explicit missing assembly paths so the result is identical on a
    development machine and on the camera machine: only the assembly location is probed here, so no
    pythonnet/.NET runtime and no driver is loaded by this check.
    """

    from devices.factory import create_ccd_devices
    from devices.lsi8181 import DLL_PATH_ENV as LSI_DLL_PATH_ENV
    from devices.sapera_api import DLL_PATH_ENV as SAPERA_DLL_PATH_ENV

    with tempfile.TemporaryDirectory(prefix="visionflow_packaged_ccd_") as temporary:
        devices = create_ccd_devices(
            {
                SAPERA_DLL_PATH_ENV: str(Path(temporary) / "SapClassBasic.dll"),
                LSI_DLL_PATH_ENV: str(Path(temporary) / "LSI8181_64.dll"),
            }
        )
        try:
            camera = devices.camera.availability()
            meter_wheel = devices.meter_wheel.availability()
        finally:
            devices.close()
    if camera.available or meter_wheel.available:
        return 13
    if "E-0201" not in camera.reason or not meter_wheel.reason:
        return 14

    from PySide6.QtWidgets import QApplication

    from gui.screens.ccd_screen import CcdScreen

    app = QApplication.instance() or QApplication([])
    screen = CcdScreen()
    screen.set_availability(camera, meter_wheel)
    app.processEvents()
    shown = (
        not screen.camera_availability_label.isHidden()
        and camera.reason in screen.camera_availability_label.text()
        and not screen.meter_wheel_availability_label.isHidden()
    )
    screen.deleteLater()
    app.processEvents()
    return 0 if shown else 15


def _packaged_smoke_recipe() -> dict:
    return {
        "recipe_name": "PACKAGED_GPU_FALLBACK_SMOKE",
        "product_id": "SMOKE",
        "machine_id": "SMOKE",
        "version": "1.0.0",
        "gpu": {
            "mode": "cpu",
            "tiling": False,
            "display": False,
            "dll_path": "missing.dll",
            "fallback_to_cpu": True,
        },
        "tile": {"mode": "grid", "width": 64, "height": 64, "overlap_x": 0, "overlap_y": 0},
        "decision": {
            "mode": "all_detectors_must_pass",
            "important_detectors": ["401-CS-AP-1"],
            "max_ng_count": 0,
        },
        "detectors": {
            "401-CS-AP-1": {
                "enabled": True,
                "use_gpu": False,
                "display_name": "packaged fallback smoke",
                "params": {
                    "blur_size": 3,
                    "adaptive_block_size": 3,
                    "adaptive_c": -2.0,
                    "roi_inset_px": 0,
                    "contour_mode": "external",
                    "morph_operation": "none",
                    "process_scale": 1.0,
                    "min_area": 0,
                    "max_area": 0,
                    "min_circularity": 0,
                    "min_fill_ratio": 0,
                    "max_fill_ratio": 0,
                },
            }
        },
        "output": {
            "save_overlay": False,
            "save_ng_tiles": False,
            "save_csv": False,
            "save_matrix_csv": False,
            "save_json": False,
        },
    }


def _normalized_smoke_result(result: dict) -> dict:
    normalized = deepcopy(result)
    for key in ("duration_sec", "outputs", "execution", "provenance"):
        normalized.pop(key, None)
    for tile_result in normalized["tiles"]:
        for detector_result in tile_result["detectors"]:
            detector_result.pop("execution", None)
    return normalized


def run_packaged_gpu_fallback_smoke_test() -> int:
    """Run a small packaged pipeline matrix for missing-DLL fallback on and off."""
    import cv2
    import numpy as np
    import yaml

    from core.gpu_runtime import GpuRuntimeError
    from core.pipeline import AOIPipeline

    with tempfile.TemporaryDirectory(prefix="visionflow_packaged_smoke_") as temporary:
        root = Path(temporary)
        image_path = root / "input.png"
        image = np.random.default_rng(20260717).integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
        encoded, payload = cv2.imencode(".png", image)
        if not encoded:
            return 4
        image_path.write_bytes(payload.tobytes())

        cpu_recipe = _packaged_smoke_recipe()
        fallback_recipe = deepcopy(cpu_recipe)
        fallback_recipe["gpu"].update(
            mode="auto",
            tiling=True,
            dll_path=str(root / "definitely_missing.dll"),
            fallback_to_cpu=True,
        )
        fallback_recipe["detectors"]["401-CS-AP-1"]["use_gpu"] = True
        strict_recipe = deepcopy(fallback_recipe)
        strict_recipe["gpu"].update(mode="cuda", fallback_to_cpu=False)

        paths = {}
        for name, recipe in (
            ("cpu", cpu_recipe),
            ("fallback", fallback_recipe),
            ("strict", strict_recipe),
        ):
            paths[name] = root / f"{name}.yaml"
            paths[name].write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")

        cpu_result = AOIPipeline(paths["cpu"], root / "cpu_output").run(image_path)
        fallback_result = AOIPipeline(paths["fallback"], root / "fallback_output").run(image_path)
        if _normalized_smoke_result(cpu_result) != _normalized_smoke_result(fallback_result):
            return 5
        gpu_report = fallback_result.get("execution", {}).get("gpu", {})
        if gpu_report.get("metrics", {}).get("call_count") != 0:
            return 6
        try:
            AOIPipeline(paths["strict"], root / "strict_output").run(image_path)
        except GpuRuntimeError as exc:
            if "CUDA DLL not found" not in str(exc):
                return 7
        else:
            return 8
    return 0


def run_packaged_yolox_smoke_test() -> int:
    """Verify the bundled registry, ONNX fixture, and CPU YOLOX pipeline."""
    import cv2
    import numpy as np

    from core.ai_runtime import AiModelError, YoloXModelRegistry
    from core.pipeline import AOIPipeline

    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    recipe_path = (
        bundle_root
        / "recipes"
        / "examples"
        / "YOLOX_TINY_REFERENCE_AOI_01.yaml"
    )
    try:
        registry = YoloXModelRegistry()
        manifest = registry.get("yolox_tiny_fixture")
    except AiModelError:
        return 9
    if not recipe_path.is_file() or not manifest.model_path.is_file():
        return 10

    with tempfile.TemporaryDirectory(prefix="visionflow_packaged_yolox_") as temporary:
        root = Path(temporary)
        image_path = root / "input.png"
        encoded, payload = cv2.imencode(
            ".png", np.zeros((32, 32, 3), dtype=np.uint8)
        )
        if not encoded:
            return 11
        image_path.write_bytes(payload.tobytes())
        result = AOIPipeline(recipe_path, root / "output").run(image_path)

    defect_types = [
        defect["type"]
        for tile in result["tiles"]
        for detector in tile["detectors"]
        if detector["detector_id"] == "yolox"
        for defect in detector["defects"]
    ]
    ai = result.get("execution", {}).get("ai", {})
    if (
        result.get("final_result") != "NG"
        or defect_types != ["scratch", "stain"]
        or ai.get("load_count") != 1
        or ai.get("session_count") != 1
        or ai.get("sessions", [{}])[0].get("backend") != "onnxruntime_cpu"
    ):
        return 12
    return 0


def sapera_diagnose_text(report) -> str:
    """The manually-copyable diagnosis text: step summary, one short line per step, report paths."""

    lines = [report.summary(), ""]
    lines.extend(report.lines())
    lines.extend(["", f"完整報告：{report.report_path}", f"機器可讀報告：{report.log_path}"])
    return "\n".join(lines)


def build_sapera_diagnose_dialog(report):
    """Read-only dialog used because a windowed package has no console to print the short codes to."""

    from PySide6.QtWidgets import QDialog, QDialogButtonBox, QPlainTextEdit, QVBoxLayout

    dialog = QDialog()
    dialog.setWindowTitle("Sapera 相機診斷")
    view = QPlainTextEdit()
    view.setObjectName("sapera_diagnose_text")
    view.setReadOnly(True)
    view.setPlainText(sapera_diagnose_text(report))
    layout = QVBoxLayout(dialog)
    layout.addWidget(view)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dialog.reject)
    buttons.accepted.connect(dialog.accept)
    layout.addWidget(buttons)
    dialog.resize(760, 420)
    return dialog


def run_packaged_sapera_diagnose(runner=None, *, show_dialog: bool = True) -> int:
    """Field diagnosis entry for the packaged (windowed, console-less) executable.

    `main.py --sapera-diagnose` prints the same short codes when a console exists; the packaged EXE
    shows them in a dialog instead. `show_dialog=False` is the test hook.
    """

    from devices.sapera_diagnose import run_sapera_diagnose

    report = (runner or run_sapera_diagnose)()
    if show_dialog:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        dialog = build_sapera_diagnose_dialog(report)
        dialog.exec()
    return 0 if report.passed else 1


if __name__ == "__main__":
    if "--smoke-test" in sys.argv[1:]:
        raise SystemExit(run_packaged_smoke_test())
    if "--sapera-diagnose" in sys.argv[1:]:
        raise SystemExit(run_packaged_sapera_diagnose())
    raise SystemExit(run_app())
