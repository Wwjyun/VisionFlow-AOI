"""Timing evidence for the optional float32 Gaussian export against cv2.GaussianBlur.

Times ``vf_gaussian_blur_f32`` and the OpenCV reference on the same 4000x2000 float32 plane, and
reports the CUDA-event breakdown (h2d / kernel / d2h / total device) that the persistent context
records for the device call. The kernel-only row is the floor a fully resident pipeline could reach
once the residual stops crossing PCIe.

Run with the repository Python:

    .\\env\\Scripts\\python.exe tools\\gaussian_f32_timing.py
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gpu_runtime import GpuRuntime  # noqa: E402

EVIDENCE_DIR = ROOT / "outputs_validation" / "cnr_profile"
EVIDENCE_TEXT = EVIDENCE_DIR / "gaussian_f32_timing.txt"
EVIDENCE_JSON = EVIDENCE_DIR / "gaussian_f32_timing.json"
ROI_IMAGE = EVIDENCE_DIR / "roi_2000x4000.png"
KERNEL_SIZE = 51


def load_roi() -> np.ndarray:
    """The profiled 4000x2000 ROI as the detector sees it: uint8 gray widened to float32."""
    if ROI_IMAGE.is_file():
        image = cv2.imread(str(ROI_IMAGE), cv2.IMREAD_UNCHANGED)
        if image is not None:
            gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if gray.shape == (2000, 4000):
                return gray.astype(np.float32)
    rng = np.random.default_rng(20260914)
    base = np.full((2000, 4000), 150, dtype=np.float32)
    base += rng.normal(0.0, 2.0, base.shape).astype(np.float32)
    base[900:1100, 1900:2100] = 40.0
    return base


def summarize(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "min_ms": ordered[0],
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))],
        "mean_ms": statistics.fmean(ordered),
        "samples": len(ordered),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", default=str(ROOT / "gpu" / "visionflow_cuda.dll"))
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    runtime = GpuRuntime(args.dll)
    if not runtime.available or not runtime.supports_gaussian_blur_f32:
        print(f"CUDA float32 Gaussian unavailable: {runtime.unavailable_reason or 'missing export'}")
        return 2

    source = load_roi()
    height, width = source.shape
    print(f"ROI {width}x{height} float32 ({source.nbytes / (1 << 20):.1f} MiB), ksize={KERNEL_SIZE}")

    # Warm up both paths (JIT, page faults, first-touch of the grow-only device buffers).
    for _ in range(3):
        cv2.GaussianBlur(source, (KERNEL_SIZE, KERNEL_SIZE), 0.0)
        runtime.gaussian_blur_f32(source, KERNEL_SIZE)

    cv2_samples: list[float] = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        reference = cv2.GaussianBlur(source, (KERNEL_SIZE, KERNEL_SIZE), 0.0)
        cv2_samples.append((time.perf_counter() - started) * 1000.0)

    device_samples: list[float] = []
    event_rows: list[dict] = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        actual = runtime.gaussian_blur_f32(source, KERNEL_SIZE)
        device_samples.append((time.perf_counter() - started) * 1000.0)
        timings = runtime.performance_stats()["native_timings_ms"] or {}
        event_rows.append({
            "h2d_ms": float(timings.get("h2d_ms", 0.0)),
            "kernel_ms": float(timings.get("kernel_ms", 0.0)),
            "d2h_ms": float(timings.get("d2h_ms", 0.0)),
            "total_device_ms": float(timings.get("total_device_ms", 0.0)),
        })

    difference = np.abs(actual.astype(np.float64) - reference.astype(np.float64))

    cv2_stats = summarize(cv2_samples)
    device_stats = summarize(device_samples)
    event_stats = {
        key: summarize([row[key] for row in event_rows])
        for key in ("h2d_ms", "kernel_ms", "d2h_ms", "total_device_ms")
    }
    speedup = cv2_stats["median_ms"] / device_stats["median_ms"]

    lines = [
        "== vf_gaussian_blur_f32 timing on the profiled 4000x2000 ROI ==",
        f"device: {runtime.device_name} (sm {runtime.compute_capability})",
        f"operand: float32 {width}x{height}, {source.nbytes / (1 << 20):.1f} MiB, ksize={KERNEL_SIZE}, "
        f"repeats={args.repeats} after 3 warm-ups",
        f"max|device - cv2| = {difference.max():.3e}  mean = {difference.mean():.3e}",
        "",
        "[host wall clock]",
        f"  cv2.GaussianBlur           median {cv2_stats['median_ms']:8.3f} ms  "
        f"min {cv2_stats['min_ms']:8.3f}  p95 {cv2_stats['p95_ms']:8.3f}",
        f"  runtime.gaussian_blur_f32  median {device_stats['median_ms']:8.3f} ms  "
        f"min {device_stats['min_ms']:8.3f}  p95 {device_stats['p95_ms']:8.3f}",
        f"  speedup (median)           {speedup:8.2f} x",
        "",
        "[cuda events inside the device call]",
    ]
    for key, label in (
        ("h2d_ms", "h2d (upload)"),
        ("kernel_ms", "kernel (h+v)"),
        ("d2h_ms", "d2h (download)"),
        ("total_device_ms", "device total"),
    ):
        stats = event_stats[key]
        lines.append(
            f"  {label:16s} median {stats['median_ms']:8.3f} ms  min {stats['min_ms']:8.3f}  "
            f"p95 {stats['p95_ms']:8.3f}"
        )
    kernel_fraction = event_stats["kernel_ms"]["median_ms"] / device_stats["median_ms"] * 100.0
    lines.append(
        f"  kernel share of the host-visible call: {kernel_fraction:.1f} % "
        "(the remainder is the two 32 MiB PCIe transfers plus launch overhead)"
    )
    lines.append("")
    lines.append("[residency note]")
    lines.append(
        "  A resident pipeline that keeps the residual on the device would replace the "
        f"median {device_stats['median_ms']:.3f} ms call with about "
        f"{event_stats['kernel_ms']['median_ms']:.3f} ms of kernel time "
        f"(saving the {event_stats['h2d_ms']['median_ms']:.3f} ms upload and the "
        f"{event_stats['d2h_ms']['median_ms']:.3f} ms download)."
    )

    record = {
        "device": runtime.device_name,
        "compute_capability": runtime.compute_capability,
        "shape": [height, width],
        "kernel_size": KERNEL_SIZE,
        "cpu_reference_max_abs": float(difference.max()),
        "cpu_reference_mean_abs": float(difference.mean()),
        "cv2": cv2_stats,
        "device": device_stats,
        "cuda_events": event_stats,
        "speedup_median": speedup,
    }
    EVIDENCE_TEXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    EVIDENCE_JSON.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {EVIDENCE_TEXT.relative_to(ROOT)}")
    print(f"written: {EVIDENCE_JSON.relative_to(ROOT)}")
    runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
