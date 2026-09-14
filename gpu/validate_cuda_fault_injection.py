"""Inject real CUDA failures on an NVIDIA device and verify CPU fallback/recovery.

Scenarios (no fake DLL):
- init_failure: child process with CUDA_VISIBLE_DEVICES=-1 (driver reports no device).
- kernel_launch_error: a 1-pixel-wide image taller than 65535 * BLOCK_Y rows makes the
  CUDA kernel grid invalid, so the launch itself fails inside the real DLL.
- device_oom: a ROI batch whose device allocation exceeds dedicated + shared GPU memory.
- vram_pressure (opt-in): a separate process holds the free dedicated VRAM while plans run.

Each scenario checks CPU-equivalent results, whole-detector fallback metadata and that the
same runtime/session recovers afterwards without stale errors, handles or pointers.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.detector_manager import DetectorManager  # noqa: E402
from core.gpu_runtime import GpuRuntime, GpuRuntimeError  # noqa: E402
from core.gpu_session import GpuExecutionSession  # noqa: E402
from core.pipeline import AOIPipeline  # noqa: E402
from core.preprocess_plan import (  # noqa: E402
    AdaptiveMean, CpuPreprocessExecutor, Gaussian, Gray, Morphology, PreprocessPlan,
)

DETECTOR_ID = "505-AS-SN-1"
# BLOCK_Y (16) * 65535 grid rows < TALL_HEIGHT <= OpenCV's default 1 << 20 decode limit.
TALL_HEIGHT = 1_048_570
DISABLED_OUTPUT = {
    "save_overlay": False, "save_ng_tiles": False, "save_csv": False,
    "save_matrix_csv": False, "save_json": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dll", default="gpu/visionflow_cuda.dll")
    parser.add_argument("--vram-pressure", action="store_true",
                        help="Also hold free dedicated VRAM from another process while plans run.")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--json-output")
    parser.add_argument("--child", choices=["init-failure", "vram-hog"], help=argparse.SUPPRESS)
    return parser.parse_args()


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _without_execution(result: dict) -> dict:
    stripped = deepcopy(result)
    stripped.pop("execution", None)
    return stripped


def _normalized_pipeline(result: dict) -> dict:
    normalized = deepcopy(dict(result))
    for key in ("duration_sec", "outputs", "execution", "provenance"):
        normalized.pop(key, None)
    for tile in normalized.get("tiles", []):
        for detector in tile.get("detectors", []):
            detector.pop("execution", None)
    return normalized


def _recipe(gpu: dict, tile: dict, use_gpu: bool) -> dict:
    return {
        "recipe_name": "CUDA_FAULT_INJECTION", "product_id": "VALIDATION", "machine_id": "RTX",
        "version": "0.1.0", "gpu": gpu, "tile": tile,
        "decision": {"mode": "all_detectors_must_pass", "important_detectors": [DETECTOR_ID], "max_ng_count": 0},
        "detectors": {DETECTOR_ID: {"enabled": True, "use_gpu": use_gpu, "params": {}}},
        "output": dict(DISABLED_OUTPUT),
    }


def _write_recipe(path: Path, recipe: dict) -> Path:
    path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")
    return path


def _write_image(path: Path, image: np.ndarray) -> Path:
    if not cv2.imwrite(str(path), image):
        raise AssertionError(f"Failed to write validation image: {path}")
    return path


def _synthetic_bgr(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((height, width, 3), 180, dtype=np.uint8)
    noise = rng.integers(0, 30, (height, width, 1), dtype=np.uint8)
    image = cv2.subtract(image, np.repeat(noise, 3, axis=2))
    for _ in range(4):
        x, y = int(rng.integers(8, max(9, width - 40))), int(rng.integers(8, max(9, height - 40)))
        cv2.rectangle(image, (x, y), (x + 24, y + 18), (20, 20, 20), -1)
    return image


def scenario_init_failure(dll: str) -> dict:
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child", "init-failure", "--dll", dll],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=600,
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if completed.returncode != 0 or not lines:
        raise AssertionError(f"init-failure child failed ({completed.returncode}): {completed.stderr[-2000:]}")
    payload = json.loads(lines[-1])
    print(f"PASS init_failure: {payload}")
    return payload


def child_init_failure(dll: str) -> dict:
    runtime = GpuRuntime(dll, fallback_to_cpu=True)
    _check(not runtime.available, "CUDA_VISIBLE_DEVICES=-1 still exposed a CUDA device")
    image = _synthetic_bgr(512, 640, seed=11)
    manager = DetectorManager()
    cpu = manager.create(DETECTOR_ID, use_gpu=False).run(image)
    fallback = manager.create(DETECTOR_ID, use_gpu=True, gpu_runtime=runtime).run(image)
    _check(_without_execution(fallback) == _without_execution(cpu), "Detector fallback differs from CPU")
    _check(fallback["execution"]["backend"] == "cpu" and fallback["execution"]["fallback_reason"],
           "Detector did not report CPU fallback reason")

    tile = {"mode": "grid", "width": 256, "height": 256, "overlap_x": 0, "overlap_y": 0}
    with tempfile.TemporaryDirectory(prefix="visionflow_fault_init_") as temporary:
        folder = Path(temporary)
        image_path = _write_image(folder / "input.png", image)
        cpu_recipe = _write_recipe(folder / "cpu.yaml", _recipe({"mode": "cpu", "dll_path": dll}, tile, False))
        auto_recipe = _write_recipe(folder / "auto.yaml", _recipe(
            {"mode": "auto", "dll_path": dll, "fallback_to_cpu": True, "tiling": True}, tile, True))
        strict_recipe = _write_recipe(folder / "strict.yaml", _recipe(
            {"mode": "cuda", "dll_path": dll, "tiling": True}, tile, True))
        cpu_result = AOIPipeline(cpu_recipe, folder / "cpu").run(image_path)
        auto_result = AOIPipeline(auto_recipe, folder / "auto").run(image_path)
        _check(_normalized_pipeline(auto_result) == _normalized_pipeline(cpu_result),
               "auto-mode CPU fallback pipeline differs from CPU mode")
        auto_gpu = auto_result["execution"]["gpu"]
        _check(auto_gpu["metrics"]["call_count"] == 0, "auto fallback still issued CUDA calls")
        _check(not auto_gpu["detectors"][DETECTOR_ID]["active"], "auto fallback reported CUDA active")
        strict_error = ""
        try:
            AOIPipeline(strict_recipe, folder / "strict").run(image_path)
        except GpuRuntimeError as exc:
            strict_error = str(exc)
        _check(bool(strict_error), "strict CUDA mode did not fail without a device")
    return {
        "unavailable_reason": runtime.unavailable_reason,
        "detector_fallback_reason": fallback["execution"]["fallback_reason"],
        "pipeline_final_result": cpu_result["final_result"],
        "pipeline_tile_count": cpu_result["summary"]["tile_count"],
        "auto_gpu_calls": auto_gpu["metrics"]["call_count"],
        "strict_error": strict_error,
    }


def scenario_kernel_launch_error(dll: str) -> dict:
    manager = DetectorManager()
    normal = _synthetic_bgr(512, 640, seed=21)
    tall = np.random.default_rng(22).integers(0, 256, (TALL_HEIGHT, 1, 3), dtype=np.uint8)
    cpu_normal = _without_execution(manager.create(DETECTOR_ID).run(normal))
    cpu_tall = _without_execution(manager.create(DETECTOR_ID).run(tall))
    with GpuRuntime(dll, fallback_to_cpu=True) as runtime:
        _check(runtime.available, runtime.unavailable_reason)
        runs = []
        for label, image, expected, expect_fallback in (
            ("before", normal, cpu_normal, False),
            ("launch_error", tall, cpu_tall, True),
            ("after", normal, cpu_normal, False),
        ):
            result = manager.create(DETECTOR_ID, use_gpu=True, gpu_runtime=runtime).run(image)
            reason = result["execution"]["fallback_reason"]
            _check(_without_execution(result) == expected, f"{label}: detector result differs from CPU")
            _check(bool(reason) == expect_fallback and result["execution"]["gpu_active"] != expect_fallback,
                   f"{label}: unexpected routing active={result['execution']['gpu_active']} reason={reason!r}")
            if expect_fallback:
                _check("CUDA DLL error" in reason, f"launch error was not reported by the DLL: {reason!r}")
            runs.append({
                "run": label, "gpu_active": result["execution"]["gpu_active"], "fallback_reason": reason,
                "defects": len(result["defects"]), "context": runtime.performance_stats()["persistent_context"],
            })
        strict = GpuRuntime(dll, fallback_to_cpu=False)
        try:
            strict_error = ""
            try:
                manager.create(DETECTOR_ID, use_gpu=True, gpu_runtime=strict).run(tall)
            except Exception as exc:  # strict mode must surface the CUDA failure itself
                strict_error = str(exc)
            _check("CUDA DLL error" in strict_error, f"strict detector hid the launch error: {strict_error!r}")
        finally:
            strict.close()
        session_runs = _session_crop_recovery(dll, normal, tall)
    payload = {"detector_runs": runs, "strict_error": strict_error, "session_tiling_runs": session_runs}
    print(f"PASS kernel_launch_error: {json.dumps(payload, ensure_ascii=False)}")
    return payload


def _session_crop_recovery(dll: str, normal: np.ndarray, tall: np.ndarray) -> list[dict]:
    narrow = np.ascontiguousarray(normal[:, :1])
    tile = {"mode": "grid", "width": 1, "height": TALL_HEIGHT, "overlap_x": 0, "overlap_y": 0}
    gpu = {"mode": "auto", "dll_path": dll, "fallback_to_cpu": True, "tiling": True}
    runs = []
    with tempfile.TemporaryDirectory(prefix="visionflow_fault_session_") as temporary:
        folder = Path(temporary)
        recipe_path = _write_recipe(folder / "session.yaml", _recipe(gpu, tile, False))
        cpu_recipe = _write_recipe(folder / "cpu.yaml", _recipe({"mode": "cpu", "dll_path": dll}, tile, False))
        images = {
            "before": _write_image(folder / "narrow.bmp", narrow),
            "crop_launch_error": _write_image(folder / "tall.bmp", tall),
        }
        images["after"] = images["before"]
        session = GpuExecutionSession.from_recipe_path(recipe_path)
        try:
            for label in ("before", "crop_launch_error", "after"):
                cpu_result = AOIPipeline(cpu_recipe, folder / f"cpu_{label}").run(images[label])
                result = AOIPipeline(recipe_path, folder / label, gpu_session=session).run(images[label])
                tiling = result["execution"]["gpu"]["tiling"]
                _check(_normalized_pipeline(result) == _normalized_pipeline(cpu_result),
                       f"session {label}: pipeline differs from CPU")
                expect_error = label == "crop_launch_error"
                _check(tiling["active"] != expect_error and bool(tiling["fallback_reason"]) == expect_error,
                       f"session {label}: stale or missing tiling fallback {tiling}")
                runs.append({"run": label, "tiling_active": tiling["active"],
                             "fallback_reason": tiling["fallback_reason"],
                             "final_result": result["final_result"]})
        finally:
            session.close()
    return runs


def scenario_device_oom(dll: str) -> dict:
    image = _synthetic_bgr(4096, 4096, seed=31)
    plan = PreprocessPlan((Gray(), Gaussian(5), AdaptiveMean(31, 5.0, 255, True),
                           Morphology("open", 5, 2)), name="fault_oom_recovery")
    roi = (1024, 512, 1024, 1024)
    x, y, width, height = roi
    expected = CpuPreprocessExecutor().execute(image[y:y + height, x:x + width], plan)
    with GpuRuntime(dll, fallback_to_cpu=True) as runtime:
        _check(runtime.available and runtime.supports_roi_batch, "ROI batch exports are unavailable")
        resident = runtime.upload_image(image)
        oom_error = ""
        try:
            runtime.create_roi_batch(resident, [(0, 0, 4096, 4096)] * 65535).close()
        except GpuRuntimeError as exc:
            oom_error = str(exc)
        _check("error 1002" in oom_error, f"oversized ROI batch did not hit device OOM: {oom_error!r}")
        _check(not runtime._roi_batches, "failed ROI batch left a registered native handle")
        recovered = []
        for attempt in range(3):
            with runtime.create_roi_batch(resident, [roi, (0, 0, width, height)]) as batch:
                _check(np.array_equal(batch.download(0), image[y:y + height, x:x + width]),
                       f"ROI batch pixels differ after OOM (attempt {attempt})")
            actual = runtime.execute_plan(image[y:y + height, x:x + width], plan,
                                          device_roi=resident.roi(*roi))
            _check(np.array_equal(actual, expected), f"resident plan differs after OOM (attempt {attempt})")
            recovered.append(runtime.performance_stats()["persistent_context"])
        _check(recovered[0]["allocation_count"] == recovered[-1]["allocation_count"],
               f"allocation count kept growing after OOM recovery: {recovered}")
        memory = runtime.memory_info()
    payload = {"oom_error": oom_error, "recovered_context": recovered[-1],
               "registered_batches": 0, "memory_after": memory}
    print(f"PASS device_oom: {json.dumps(payload)}")
    return payload


def _cudart():
    return ctypes.CDLL(str(Path(os.environ["CUDA_PATH"]) / "bin" / "x64" / "cudart64_13.dll"))


def child_vram_hog() -> None:
    cudart = _cudart()
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    if cudart.cudaSetDevice(0) != 0:
        raise SystemExit("cudaSetDevice failed")
    # Hold only dedicated VRAM reported free; never allocate into shared system memory.
    held, held_bytes, chunk, reserve = [], 0, 256 << 20, 128 << 20
    while True:
        cudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
        if free.value <= reserve + chunk // 16:
            break
        size = min(chunk, free.value - reserve)
        pointer = ctypes.c_void_p()
        if cudart.cudaMalloc(ctypes.byref(pointer), ctypes.c_size_t(size)) != 0:
            break
        held.append(pointer)
        held_bytes += size
    cudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    print(json.dumps({"held_mib": held_bytes >> 20, "free_mib": free.value >> 20}), flush=True)
    sys.stdin.readline()


def scenario_vram_pressure(dll: str, repetitions: int) -> dict:
    image = _synthetic_bgr(2160, 3840, seed=41)
    plan = PreprocessPlan((Gray(), Gaussian(5), AdaptiveMean(31, 5.0, 255, True),
                           Morphology("open", 5, 2)), name="fault_vram_pressure")
    expected = CpuPreprocessExecutor().execute(image, plan)

    def timed(runtime) -> list[float]:
        samples = []
        for _ in range(repetitions):
            started = time.perf_counter()
            actual = runtime.execute_plan(image, plan)
            samples.append((time.perf_counter() - started) * 1000.0)
            _check(np.array_equal(actual, expected), "native plan differs from CPU under VRAM pressure")
        return samples

    with GpuRuntime(dll, fallback_to_cpu=False) as runtime:
        runtime.execute_plan(image, plan)
        baseline = timed(runtime)
        hog = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--child", "vram-hog"],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        try:
            held = json.loads(hog.stdout.readline())
            fresh = GpuRuntime(dll, fallback_to_cpu=False)
            try:
                pressured_new_context = timed(fresh)
                fresh_context = fresh.performance_stats()["persistent_context"]
            finally:
                fresh.close()
            pressured_warm_context = timed(runtime)
        finally:
            hog.stdin.write("\n")
            hog.stdin.flush()
            hog.wait(timeout=120)
        recovered = timed(runtime)

    def summary(samples):
        return {"median_ms": round(statistics.median(samples), 3), "max_ms": round(max(samples), 3)}

    payload = {
        "hog": held,
        "baseline": summary(baseline),
        "pressured_new_context": summary(pressured_new_context),
        "pressured_new_context_reserved_bytes": fresh_context["reserved_bytes"],
        "pressured_warm_context": summary(pressured_warm_context),
        "recovered": summary(recovered),
    }
    print(f"PASS vram_pressure: {json.dumps(payload)}")
    return payload


def main() -> int:
    args = parse_args()
    if args.child == "init-failure":
        print(json.dumps(child_init_failure(args.dll), ensure_ascii=False))
        return 0
    if args.child == "vram-hog":
        child_vram_hog()
        return 0
    probe = GpuRuntime(args.dll, fallback_to_cpu=False)
    if not probe.available:
        raise SystemExit(f"CUDA DLL unavailable: {probe.unavailable_reason}")
    device = {"name": probe.device_name, "compute_capability": probe.compute_capability}
    probe.close()
    report = {
        "schema_version": 1,
        "device": device,
        "init_failure": scenario_init_failure(args.dll),
        "kernel_launch_error": scenario_kernel_launch_error(args.dll),
        "device_oom": scenario_device_oom(args.dll),
    }
    if args.vram_pressure:
        report["vram_pressure"] = scenario_vram_pressure(args.dll, max(1, args.repetitions))
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PASS CUDA fault injection: init failure, kernel launch error and device OOM recovered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
