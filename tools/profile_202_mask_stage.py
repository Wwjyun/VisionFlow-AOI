"""Confirm the wired-in GPU median is actually used, and split the 202 mask stage.

Two questions this answers with evidence rather than inference:

1. ``Detector202_1._exact_median`` silently falls back to ``np.median`` when the
   runtime is missing the export or ``use_gpu`` is false.  A timing win alone
   does not prove the device path ran, so this counts the calls the detector
   makes into the runtime.
2. ``automatic_cnr_mask`` is the only remaining hot stage; this splits it into
   Gaussian background, the two exact medians, the threshold/mask construction
   and the morphology, so the next GPU target is chosen from measurement.

Usage:
    .\\env\\Scripts\\python.exe tools\\profile_202_mask_stage.py [--json OUTPUT]
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


class CountingRuntime:
    """Wrap a GpuRuntime and count every median the detector asks the device for."""

    def __init__(self, runtime: GpuRuntime):
        self._runtime = runtime
        self.median_calls = 0
        self.median_values = 0

    def __getattr__(self, name):
        return getattr(self._runtime, name)

    def median_f32(self, values: np.ndarray) -> np.float32:
        self.median_calls += 1
        self.median_values += int(np.asarray(values).size)
        return self._runtime.median_f32(values)


def _measure_once(detector: Detector202_1, roi: np.ndarray) -> dict:
    timings: dict[str, float] = {}
    gray = detector._make_gray(roi)
    started = time.perf_counter()
    image_float = gray.astype(np.float32)
    height, width = gray.shape[:2]
    background_kernel = detector._background_kernel(height, width)
    gaussian_sigma = float(detector.params.get("gaussian_sigma", 0.0))
    timings["gray_convert"] = time.perf_counter() - started

    started = time.perf_counter()
    background = cv2.GaussianBlur(
        image_float, (background_kernel, background_kernel), gaussian_sigma
    )
    residual = image_float - background
    timings["gaussian_and_residual"] = time.perf_counter() - started

    started = time.perf_counter()
    residual_median = detector._exact_median(residual)
    timings["median_residual"] = time.perf_counter() - started

    started = time.perf_counter()
    mad = detector._exact_median(np.abs(residual - residual_median))
    timings["median_mad"] = time.perf_counter() - started

    started = time.perf_counter()
    mad_scale = float(detector.params.get("mad_scale", 1.4826))
    noise_sigma_floor = float(detector.params.get("noise_sigma_floor", 0.000001))
    robust_noise_sigma = float(max(mad_scale * mad, noise_sigma_floor))
    residual_threshold_floor = float(
        detector.params.get("residual_threshold_floor", 8.0)
    )
    residual_sigma_multiplier = float(
        detector.params.get("residual_sigma_multiplier", 3.0)
    )
    residual_threshold = float(
        max(
            residual_threshold_floor,
            residual_sigma_multiplier * robust_noise_sigma,
        )
    )
    candidate_max_value = int(detector.params.get("candidate_max_value", 255))
    candidate_mask = (
        (np.abs(residual - residual_median) > residual_threshold).astype(np.uint8)
        * candidate_max_value
    )
    timings["threshold_and_mask"] = time.perf_counter() - started

    started = time.perf_counter()
    morph_operation = str(detector.params.get("morph_operation", "open")).lower()
    morph_kernel = int(detector.params.get("morph_kernel", 3))
    morph_iterations = int(detector.params.get("morph_iterations", 1))
    cv_morphology = detector._MORPH_OPERATIONS.get(morph_operation)
    if cv_morphology is not None and morph_iterations > 0 and morph_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_kernel, morph_kernel)
        )
        candidate_mask = cv2.morphologyEx(
            candidate_mask, cv_morphology, kernel, iterations=morph_iterations
        )
    inclusion_mask = detector._apply_exclusion_masks(
        np.full(gray.shape, 255, dtype=np.uint8)
    )
    candidate_mask = cv2.bitwise_and(candidate_mask, inclusion_mask)
    timings["morphology_and_inclusion"] = time.perf_counter() - started

    started = time.perf_counter()
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_mask, connectivity=int(detector.params.get("connectivity", 8))
    )
    timings["connected_components"] = time.perf_counter() - started

    started = time.perf_counter()
    candidates = detector._collect_candidates(
        image_float, candidate_mask, inclusion_mask.astype(bool)
    )
    timings["collect_candidates_ring_cnr"] = time.perf_counter() - started

    return {
        "timings": timings,
        "background_kernel": background_kernel,
        "component_count": int(count),
        "candidate_count": len(candidates),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, default=Path("gpu/visionflow_cuda.dll"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    runtime = GpuRuntime(dll_path=args.dll, fallback_to_cpu=False)
    if not runtime.available or not runtime.supports_exact_median:
        print("CUDA exact median unavailable; nothing to attribute")
        return 1
    print(f"CUDA: {runtime.device_name}")

    height, width, defects = 2000, 12000, 96
    roi = _scene(height, width, defects)

    results = {}
    for label, use_gpu in (("cpu", False), ("gpu", True)):
        detector = Detector202_1(params=dict(PARAMS))
        counter = CountingRuntime(runtime)
        detector.gpu_runtime = counter
        detector.use_gpu = use_gpu
        _measure_once(detector, roi)  # warm
        runs = [_measure_once(detector, roi) for _ in range(args.repeats)]
        merged = {}
        for key in runs[0]["timings"]:
            merged[key] = statistics.median(run["timings"][key] for run in runs) * 1000.0
        results[label] = {
            "timings_ms": merged,
            "median_calls": counter.median_calls,
            "median_values": counter.median_values,
            "background_kernel": runs[0]["background_kernel"],
            "component_count": runs[0]["component_count"],
            "candidate_count": runs[0]["candidate_count"],
        }
        results[label]["mask_total_ms"] = sum(merged.values())

    print(f"ROI {height}x{width}, background kernel {results['cpu']['background_kernel']}")
    print(f"{'stage':32s} {'CPU ms':>10s} {'GPU-median ms':>14s}")
    for key in results["cpu"]["timings_ms"]:
        print(
            f"{key:32s} {results['cpu']['timings_ms'][key]:10.2f} "
            f"{results['gpu']['timings_ms'][key]:14.2f}"
        )
    print(
        f"{'TOTAL':32s} {results['cpu']['mask_total_ms']:10.2f} "
        f"{results['gpu']['mask_total_ms']:14.2f}"
    )
    print()
    print(
        "runtime median calls: cpu "
        f"{results['cpu']['median_calls']} ({results['cpu']['median_values']:,} values), "
        f"gpu {results['gpu']['median_calls']} ({results['gpu']['median_values']:,} values)"
    )
    if results["gpu"]["median_calls"] < 2 * args.repeats:
        print(
            "WARNING: the GPU run did not reach the device median twice per repeat; "
            "the speedup cannot be attributed to vf_median_f32"
        )
        return 1
    print(
        f"components: {results['cpu']['component_count']}, "
        f"candidates: {results['cpu']['candidate_count']}"
    )

    payload = {
        "kind": "profile_202_mask_stage",
        "device": runtime.device_name,
        "shape": [height, width],
        "repeats": args.repeats,
        "results": results,
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
