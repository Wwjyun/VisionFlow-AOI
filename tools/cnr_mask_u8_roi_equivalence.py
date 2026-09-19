"""Validate the resident 202 CNR export against the established host-operand GPU path.

The resident export must be byte/bit exact against ``vf_gaussian_blur_f32`` followed by
``vf_cnr_mask_f32`` for the same OpenCV uint8 gray input. Its mask must also equal the CPU OpenCV
reference. Evidence is written under ``outputs_validation/cnr_profile``.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gpu_runtime import GpuRuntime  # noqa: E402

OUT_DIR = ROOT / "outputs_validation" / "cnr_profile"
PARAMS = dict(
    sigma_multiplier=3.0,
    threshold_floor=8.0,
    absolute_floor=1e-6,
    mad_scale=1.4826,
    candidate_value=255,
)


def cpu_reference(gray: np.ndarray, kernel: int, sigma: float) -> dict:
    image = gray.astype(np.float32)
    background = cv2.GaussianBlur(image, (kernel, kernel), sigma)
    residual = image - background
    median = np.median(residual)
    mad = np.median(np.abs(residual - median))
    robust = max(PARAMS["mad_scale"] * float(mad), PARAMS["absolute_floor"])
    threshold = max(PARAMS["threshold_floor"], PARAMS["sigma_multiplier"] * robust)
    mask = (np.abs(residual - median) > threshold).astype(np.uint8) * PARAMS["candidate_value"]
    return {"residual_median": median, "mad": mad, "threshold": threshold, "mask": mask}


def gray_from(source: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) if source.ndim == 3 else source


def make_scene(height: int, width: int, channels: int, seed: int) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    base = ((x * 7 + y * 3 + (x // 19) * 31 + (y // 23) * 17 + seed) % 256).astype(np.uint8)
    if channels == 1:
        return base
    return np.dstack((base, np.roll(base, 3, axis=1), np.roll(base, 5, axis=0)))


def scalar_exact(left, right) -> bool:
    return np.float32(left).tobytes() == np.float32(right).tobytes()


def run_case(runtime: GpuRuntime, parent: np.ndarray, box, kernel: int, sigma: float) -> dict:
    x, y, width, height = box
    host_roi = parent[y : y + height, x : x + width]
    gray = gray_from(host_roi)
    image = gray.astype(np.float32)
    background = runtime.gaussian_blur_f32(image, kernel, sigma)
    chained = runtime.cnr_mask_f32(image, background, **PARAMS)
    resident = runtime.upload_image(parent).roi(x, y, width, height)
    started = time.perf_counter()
    fused = runtime.cnr_mask_u8_roi(
        resident, kernel_size=kernel, sigma=sigma, **PARAMS
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    cpu = cpu_reference(gray, kernel, sigma)
    exact_chain = (
        scalar_exact(fused["residual_median"], chained["residual_median"])
        and scalar_exact(fused["mad"], chained["mad"])
        and fused["threshold"] == chained["threshold"]
        and np.array_equal(fused["mask"], chained["mask"])
    )
    return {
        "parent_shape": list(parent.shape),
        "roi": list(box),
        "kernel": kernel,
        "sigma": sigma,
        "resident_ms": elapsed_ms,
        "resident_equals_chained_gpu": exact_chain,
        "resident_equals_cpu_mask": bool(np.array_equal(fused["mask"], cpu["mask"])),
        "cpu_mask_differing_pixels": int(np.count_nonzero(fused["mask"] != cpu["mask"])),
        "cpu_median_abs_diff": abs(float(fused["residual_median"]) - float(cpu["residual_median"])),
        "cpu_mad_abs_diff": abs(float(fused["mad"]) - float(cpu["mad"])),
        "cpu_threshold_abs_diff": abs(float(fused["threshold"]) - float(cpu["threshold"])),
    }


def main() -> int:
    runtime = GpuRuntime(ROOT / "gpu" / "visionflow_cuda.dll", fallback_to_cpu=False)
    if not runtime.available or not runtime.supports_cnr_mask_u8_roi:
        raise SystemExit("vf_cnr_mask_u8_roi is unavailable; rebuild the CUDA DLL")
    cases = []
    try:
        for channels in (1, 3):
            parent = make_scene(112, 144, channels, 17 + channels)
            for kernel in (3, 31, 51):
                for sigma in (0.0, 1.25):
                    cases.append(run_case(runtime, parent, (23, 19, 80, 64), kernel, sigma))
        production = make_scene(12000, 2000, 3, 20260915)
        for _ in range(3):
            cases.append(run_case(runtime, production, (0, 0, 2000, 12000), 51, 0.0))
        metrics = runtime.performance_stats()
    finally:
        runtime.close()

    passed = all(
        row["resident_equals_chained_gpu"] and row["resident_equals_cpu_mask"] for row in cases
    )
    payload = {
        "passed": passed,
        "case_count": len(cases),
        "chained_gpu_exact_count": sum(row["resident_equals_chained_gpu"] for row in cases),
        "cpu_mask_exact_count": sum(row["resident_equals_cpu_mask"] for row in cases),
        "worst_cpu_median_abs_diff": max(row["cpu_median_abs_diff"] for row in cases),
        "worst_cpu_mad_abs_diff": max(row["cpu_mad_abs_diff"] for row in cases),
        "worst_cpu_threshold_abs_diff": max(row["cpu_threshold_abs_diff"] for row in cases),
        "production_resident_median_ms": statistics.median(
            row["resident_ms"] for row in cases if row["roi"] == [0, 0, 2000, 12000]
        ),
        "cases": cases,
        "metrics": metrics,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "cnr_mask_u8_roi_equivalence.json"
    text_path = OUT_DIR / "cnr_mask_u8_roi_equivalence.txt"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    text_path.write_text(
        "\n".join(
            (
                f"passed={passed}",
                f"cases={payload['case_count']}",
                f"resident_equals_chained_gpu={payload['chained_gpu_exact_count']}/{payload['case_count']}",
                f"resident_equals_cpu_mask={payload['cpu_mask_exact_count']}/{payload['case_count']}",
                f"worst_cpu_median_abs_diff={payload['worst_cpu_median_abs_diff']:.9g}",
                f"worst_cpu_mad_abs_diff={payload['worst_cpu_mad_abs_diff']:.9g}",
                f"worst_cpu_threshold_abs_diff={payload['worst_cpu_threshold_abs_diff']:.9g}",
                f"production_resident_median_ms={payload['production_resident_median_ms']:.3f}",
            )
        ) + "\n",
        encoding="utf-8",
    )
    print(text_path.read_text(encoding="utf-8"), end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
