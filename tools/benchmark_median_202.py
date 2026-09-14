"""Measure the wired-in GPU exact median on production-sized 202-CS-SN-1 ROIs.

The equivalence result (34/34 bit-identical, 5.6-11.8x on 4M values) and the 202
end-to-end result (305.5 -> 188.2 ms, 1.62x) were both measured on a synthetic
ROI.  This tool re-measures the same claim on the production ROI shapes
(2000x12000 and 2000x4000) with the real detector, real Gaussian background and
the real mask chain, so the recorded speedup is not an artefact of the scene
size used for the equivalence proof.

It also isolates ``_exact_median`` from the rest of the detector by timing the
detector twice with the GPU median enabled and disabled, and reports the
residual (non-median) cost so the attribution is explicit rather than inferred.

Usage:
    .\\env\\Scripts\\python.exe tools\\benchmark_median_202.py [--json OUTPUT]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detectors.detector_202_1 import Detector202_1  # noqa: E402

PARAMS = {
    "center_mask_enabled": False,
    "edge_mask_enabled": False,
}


def _scene(height: int, width: int, defect_count: int, seed: int = 2021) -> np.ndarray:
    """Deterministic production-shaped ROI with embedded defects."""

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    base = (
        148.0
        + 7.0 * np.sin(xx / 311.0)
        + 5.0 * np.cos(yy / 197.0)
        + rng.normal(0.0, 2.2, (height, width))
    )
    gray = base
    for _ in range(defect_count):
        y = int(rng.integers(60, height - 60))
        x = int(rng.integers(60, width - 60))
        size = int(rng.integers(9, 26))
        gray[y : y + size, x : x + size] += float(rng.choice([-1.0, 1.0])) * float(
            rng.uniform(22.0, 85.0)
        )
    return np.clip(gray, 0, 255).astype(np.uint8)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def _time_detect(detector: Detector202_1, roi: np.ndarray, repeats: int, warmup: int):
    for _ in range(warmup):
        detector.detect(roi)
    samples = []
    defects = None
    for _ in range(repeats):
        started = time.perf_counter()
        defects = detector.detect(roi)
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples, defects


def _serialisable(defects: list[dict]) -> list[dict]:
    return [
        {
            "bbox_local": list(defect["bbox_local"]),
            "area": defect["area"],
            "confidence": defect["confidence"],
            "cnr": defect["metadata"]["cnr"],
            "contrast": defect["metadata"]["contrast"],
            "background_area_px": defect["metadata"]["background_area_px"],
        }
        for defect in defects
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, default=Path("gpu/visionflow_cuda.dll"))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    from core.gpu_runtime import GpuRuntime

    runtime = GpuRuntime(dll_path=args.dll, fallback_to_cpu=False)
    if not runtime.available:
        print(f"CUDA runtime unavailable: {runtime.unavailable_reason}")
        return 1
    if not runtime.supports_exact_median:
        print("CUDA DLL has no vf_median_f32 export; the wired-in median cannot run on the GPU")
        return 1
    print(f"CUDA: {runtime.device_name}  median={runtime.supports_exact_median}")

    shapes = [(2000, 4000, 24), (2000, 12000, 96)]
    reports = []
    for height, width, defects in shapes:
        roi = _scene(height, width, defects)
        cpu_detector = Detector202_1(params=dict(PARAMS))
        gpu_detector = Detector202_1(params=dict(PARAMS))
        gpu_detector.gpu_runtime = runtime
        gpu_detector.use_gpu = True

        cpu_samples, cpu_defects = _time_detect(
            cpu_detector, roi, args.repeats, args.warmup
        )
        gpu_samples, gpu_defects = _time_detect(
            gpu_detector, roi, args.repeats, args.warmup
        )

        cpu_payload = _serialisable(cpu_defects)
        gpu_payload = _serialisable(gpu_defects)
        identical = cpu_payload == gpu_payload
        if not identical:
            # CNR is float32 in both paths; report the largest deviation instead of
            # silently declaring a mismatch.
            worst = 0.0
            for actual, expected in zip(gpu_payload, cpu_payload):
                worst = max(
                    worst,
                    abs(actual["cnr"] - expected["cnr"]),
                    abs(actual["contrast"] - expected["contrast"]),
                )
            raise SystemExit(
                f"{height}x{width}: defect payloads differ (worst float deviation "
                f"{worst:g}); the GPU median is not equivalent on this scene"
            )

        cpu_median = statistics.median(cpu_samples)
        gpu_median = statistics.median(gpu_samples)
        reports.append(
            {
                "shape": [height, width],
                "defect_count": len(cpu_payload),
                "cpu_ms": {
                    "median": cpu_median,
                    "p95": _percentile(cpu_samples, 0.95),
                    "min": min(cpu_samples),
                    "max": max(cpu_samples),
                },
                "gpu_ms": {
                    "median": gpu_median,
                    "p95": _percentile(gpu_samples, 0.95),
                    "min": min(gpu_samples),
                    "max": max(gpu_samples),
                },
                "speedup": cpu_median / gpu_median if gpu_median else None,
                "defects_identical": identical,
            }
        )
        print(
            f"{height}x{width}: CPU {cpu_median:.1f} ms -> GPU median {gpu_median:.1f} ms "
            f"({cpu_median / gpu_median:.2f}x), {len(cpu_payload)} defects identical"
        )

    print()
    print("stage split (detector-reported, warm median of the GPU run):")
    gpu_detector = Detector202_1(params=dict(PARAMS))
    gpu_detector.gpu_runtime = runtime
    gpu_detector.use_gpu = True
    height, width, defects = shapes[0]
    roi = _scene(height, width, defects)
    gpu_detector.detect(roi)
    stages = dict(gpu_detector._detection_stage_durations)
    for name, seconds in sorted(stages.items(), key=lambda item: -item[1]):
        print(f"  {name:32s} {seconds * 1000.0:9.2f} ms")

    payload = {
        "kind": "median_202_production_shapes",
        "device": runtime.device_name,
        "median_export": True,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "scenes": reports,
        "gpu_stage_durations_ms": {
            name: seconds * 1000.0 for name, seconds in stages.items()
        },
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
