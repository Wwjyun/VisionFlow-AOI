"""Quantify how close the 202 mask decision sits to flipping when the Gaussian runs on the GPU.

``tools/gaussian_202_matrix.py`` shows the GPU float32 Gaussian is **bit-identical for
the candidate mask** on 39/39 scenes but leaves ``mad`` / ``residual_median`` /
``residual_threshold`` differing by about 1e-5 relative.  The mask is the decision, so
the question that matters is the margin: how far would the threshold have to move
before any candidate pixel changed side?

This tool answers it directly instead of arguing about it:

* ``mask_flip_margin`` - the smallest ``|abs(residual - residual_median) -
  residual_threshold|`` over every pixel, i.e. the distance to the nearest possible
  mask flip.  If the observed metadata deviation is orders of magnitude below this,
  the GPU Gaussian cannot change the mask on that scene.
* ``threshold_deviation`` / ``median_deviation`` - the actual metadata drift between
  the host and device backgrounds on the same scene.

Usage:
    .\\env\\Scripts\\python.exe tools\\gaussian_mask_margin.py [--json OUTPUT]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.gpu_runtime import GpuRuntime  # noqa: E402
from detectors.detector_202_1 import Detector202_1  # noqa: E402
from tools.benchmark_median_202 import PARAMS, _scene  # noqa: E402


def _scenes():
    """A spread of shapes, kernels and defect populations."""

    cases = [
        ("prod-2000x4000", 2000, 4000, 24, {}),
        ("prod-2000x12000", 2000, 12000, 96, {}),
        ("small-400x600", 400, 600, 4, {}),
        ("cliff-1000x1000", 1000, 1000, 6, {}),
        ("kernel31", 512, 512, 5, {"background_kernel_size": 31}),
        ("kernel127", 400, 600, 5, {"background_kernel_size": 127}),
        ("sigma1", 512, 512, 5, {"gaussian_sigma": 1.0}),
        ("no-morph", 512, 512, 5, {"morph_operation": "none", "morph_iterations": 0}),
        ("noise-floor", 512, 512, 5, {"noise_sigma_floor": 0.5}),
        ("single-defect", 400, 600, 1, {}),
        ("clean", 400, 600, 0, {}),
    ]
    for name, height, width, defects, extra in cases:
        params = dict(PARAMS)
        params.update(extra)
        yield name, _scene(height, width, defects), params


def _device_mask_and_threshold(
    detector: Detector202_1,
    gray: np.ndarray,
    runtime: GpuRuntime,
    use_gpu: bool,
) -> dict:
    image_float = gray.astype(np.float32)
    height, width = gray.shape[:2]
    kernel = detector._background_kernel(height, width)
    sigma = float(detector.params.get("gaussian_sigma", 0.0))
    if use_gpu:
        background = runtime.gaussian_blur_f32(image_float, kernel)
    else:
        background = cv2.GaussianBlur(image_float, (kernel, kernel), sigma)
    residual = image_float - background
    residual_median = detector._exact_median(residual)
    mad = detector._exact_median(np.abs(residual - residual_median))
    mad_scale = float(detector.params.get("mad_scale", 1.4826))
    noise_sigma_floor = float(detector.params.get("noise_sigma_floor", 0.000001))
    robust_noise_sigma = float(max(mad_scale * mad, noise_sigma_floor))
    threshold = float(
        max(
            float(detector.params.get("residual_threshold_floor", 8.0)),
            float(detector.params.get("residual_sigma_multiplier", 3.0))
            * robust_noise_sigma,
        )
    )
    deviation = np.abs(residual - residual_median)
    margin = float(np.min(np.abs(deviation - threshold)))
    mask = (
        (deviation > threshold).astype(np.uint8)
        * int(detector.params.get("candidate_max_value", 255))
    )
    return {
        "kernel": kernel,
        "residual_median": residual_median,
        "mad": mad,
        "robust_noise_sigma": robust_noise_sigma,
        "threshold": threshold,
        "mask_flip_margin": margin,
        "mask": mask,
        "background_max_abs_diff": None,
        "background": background,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, default=Path("gpu/visionflow_cuda.dll"))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    runtime = GpuRuntime(dll_path=args.dll, fallback_to_cpu=False)
    if not runtime.available or not runtime.supports_gaussian_blur_f32:
        print("CUDA float32 Gaussian unavailable")
        return 1
    print(f"CUDA: {runtime.device_name}")

    rows = []
    worst_ratio = 0.0
    for name, gray, params in _scenes():
        detector = Detector202_1(params=params)
        started = time.perf_counter()
        host = _device_mask_and_threshold(detector, gray, runtime, use_gpu=False)
        host_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        device = _device_mask_and_threshold(detector, gray, runtime, use_gpu=True)
        device_ms = (time.perf_counter() - started) * 1000.0

        background_diff = float(
            np.max(np.abs(host["background"] - device["background"]))
        )
        threshold_deviation = abs(host["threshold"] - device["threshold"])
        median_deviation = abs(
            host["residual_median"] - device["residual_median"]
        )
        mad_deviation = abs(host["mad"] - device["mad"])
        margin = min(host["mask_flip_margin"], device["mask_flip_margin"])
        mask_equal = bool(np.array_equal(host["mask"], device["mask"]))
        ratio = (
            max(threshold_deviation, median_deviation) / margin
            if margin > 0.0
            else float("inf")
        )
        worst_ratio = max(worst_ratio, ratio)
        rows.append(
            {
                "scene": name,
                "shape": list(gray.shape),
                "kernel": host["kernel"],
                "background_max_abs_diff": background_diff,
                "threshold_host": host["threshold"],
                "threshold_device": device["threshold"],
                "threshold_deviation": threshold_deviation,
                "residual_median_deviation": median_deviation,
                "mad_deviation": mad_deviation,
                "mask_flip_margin": margin,
                "deviation_to_margin_ratio": ratio,
                "mask_bit_identical": mask_equal,
                "host_gaussian_ms": host_ms,
                "device_gaussian_ms": device_ms,
            }
        )
        print(
            f"{name:18s} ksize={host['kernel']:3d} bg|diff|={background_diff:.3e} "
            f"thr|diff|={threshold_deviation:.3e} margin={margin:.3e} "
            f"ratio={ratio:.3e} mask_equal={mask_equal}"
        )

    masks_equal = all(row["mask_bit_identical"] for row in rows)
    print()
    print(f"scenes: {len(rows)}")
    print(f"candidate mask bit-identical everywhere: {masks_equal}")
    print(f"worst deviation/margin ratio: {worst_ratio:.3e}")
    print(
        "interpretation: a ratio far below 1 means the GPU Gaussian cannot change the "
        "mask on these scenes even though the metadata drifts"
    )
    print(
        f"worst background |diff|: {max(row['background_max_abs_diff'] for row in rows):.3e}"
    )
    print(
        "gaussian stage, host vs device median ms: "
        f"{statistics.median(row['host_gaussian_ms'] for row in rows):.2f} vs "
        f"{statistics.median(row['device_gaussian_ms'] for row in rows):.2f}"
    )

    payload = {
        "kind": "gaussian_mask_margin",
        "device": runtime.device_name,
        "scenes": rows,
        "masks_bit_identical": masks_equal,
        "worst_deviation_to_margin_ratio": worst_ratio,
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"wrote {args.json}")
    return 0 if masks_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
