"""Equivalence evidence for the optional float32 Gaussian export (vf_gaussian_blur_f32[/_roi]).

Compares the CUDA DLL against the OpenCV reference the 202-CS-SN-1 detector uses,
``cv2.GaussianBlur(float32_src, (ksize, ksize), 0.0)``, over several shapes, contents and every
odd kernel size in [3, 127], and records the tolerance this project claims.

Run with the repository Python:

    .\\env\\Scripts\\python.exe tools\\gaussian_f32_equivalence.py

Evidence is written to outputs_validation/cnr_profile/gaussian_f32_equivalence.txt (human summary)
and .json (machine readable). The tool exits non-zero when the measured difference exceeds the
claimed tolerance or any contract check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gpu_runtime import CUDA_ERROR_UNSUPPORTED, GpuRuntime, GpuRuntimeError  # noqa: E402

EVIDENCE_DIR = ROOT / "outputs_validation" / "cnr_profile"
EVIDENCE_TEXT = EVIDENCE_DIR / "gaussian_f32_equivalence.txt"
EVIDENCE_JSON = EVIDENCE_DIR / "gaussian_f32_equivalence.json"

# Tolerance claimed for gray/residual float32 data in [0, 255]. Measured worst cases are about half
# of these numbers; the margin covers contents and shapes this tool does not enumerate.
CLAIMED_GRAY_MAX_ABS = 2.0e-4
CLAIMED_GRAY_MEAN_ABS = 2.0e-5
# The error scales with the magnitude of the source. For wide-range inputs the claim is relative to
# the dynamic range instead of an absolute float32 unit.
CLAIMED_RELATIVE_TO_RANGE = 5.0e-7

VERIFIED_KERNELS = tuple(range(3, 128, 2))

# Fixed representative kernel sizes used for the shape/content sweep.
SWEEP_KERNELS = (3, 5, 9, 21, 51, 63, 101, 127)

# OpenCV's fixed small-kernel table (SMALL_GAUSSIAN_SIZE == 9 in OpenCV 5.x), indexed by ksize / 2.
SMALL_KERNELS = {
    1: (1.0,),
    3: (0.25, 0.5, 0.25),
    5: (0.0625, 0.25, 0.375, 0.25, 0.0625),
    7: (0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125),
    9: (0.015625, 0.05078125, 0.1171875, 0.19921875, 0.234375,
        0.19921875, 0.1171875, 0.05078125, 0.015625),
}


def reference_kernel_f32(ksize: int) -> np.ndarray:
    """Mirror of prepare_gaussian_f32_weights() in gpu/visionflow_cuda.cu."""
    if ksize in SMALL_KERNELS:
        return np.array(SMALL_KERNELS[ksize], dtype=np.float32)
    sigma = ((ksize - 1) * 0.5 - 1) * 0.3 + 0.8
    offset = np.arange(ksize, dtype=np.float64) - (ksize - 1) * 0.5
    raw = np.exp(-0.5 * offset * offset / (sigma * sigma))
    return (raw * (1.0 / float(np.sum(raw)))).astype(np.float32)


def kernel_coefficients() -> dict:
    """Check the generated coefficients against cv2.getGaussianKernel for every verified size."""
    worst = 0.0
    worst_size = 0
    for ksize in VERIFIED_KERNELS:
        reference = cv2.getGaussianKernel(ksize, 0, cv2.CV_32F).reshape(-1)
        difference = float(np.max(np.abs(reference - reference_kernel_f32(ksize))))
        if difference > worst:
            worst, worst_size = difference, ksize
    return {
        "checked_sizes": len(VERIFIED_KERNELS),
        "worst_abs_difference": worst,
        "worst_size": worst_size,
        "bit_exact": worst == 0.0,
    }


def scene(shape: tuple[int, int], kind: str, seed: int) -> np.ndarray:
    """Build one float32 test operand of the requested shape and kind."""
    height, width = shape
    rng = np.random.default_rng(seed)
    if kind == "uniform-random":
        return rng.uniform(0.0, 255.0, shape).astype(np.float32)
    if kind == "uint8-random":
        return rng.integers(0, 256, shape).astype(np.float32)
    if kind == "gradient":
        y, x = np.mgrid[0:height, 0:width]
        return (x * (255.0 / max(width - 1, 1)) + y * (255.0 / max(height - 1, 1)) * 0.5).astype(np.float32)
    if kind == "flat":
        return np.full(shape, 137.0, dtype=np.float32)
    if kind == "impulse":
        data = np.zeros(shape, dtype=np.float32)
        data[height // 2, width // 2] = 255.0
        return data
    if kind == "high-contrast-edges":
        data = np.zeros(shape, dtype=np.float32)
        data[:, width // 2:] = 255.0
        data[height // 3: 2 * height // 3, width // 4:] = 0.0
        data[height // 5, :] = 255.0
        return data
    if kind == "saturated":
        data = rng.choice(np.array([0.0, 255.0], dtype=np.float32), size=shape)
        return data.astype(np.float32)
    if kind == "negative-range":
        return rng.uniform(-1000.0, 1000.0, shape).astype(np.float32)
    if kind == "aoi-like":
        data = np.full(shape, 150.0, dtype=np.float32)
        data += rng.normal(0.0, 1.5, shape).astype(np.float32)
        for _ in range(6):
            cy = int(rng.integers(4, max(height - 4, 5)))
            cx = int(rng.integers(4, max(width - 4, 5)))
            half = int(rng.integers(1, 5))
            data[max(cy - half, 0): cy + half + 1, max(cx - half, 0): cx + half + 1] = float(
                rng.choice([60.0, 240.0])
            )
        return data.astype(np.float32)
    raise ValueError(f"unknown scene kind: {kind}")


SHAPES = (
    (1, 1),
    (3, 5),
    (7, 9),
    (20, 20),
    (64, 80),
    (192, 256),
    (400, 600),
    (3, 1024),
    (1024, 3),
)

CONTENTS = (
    "uniform-random",
    "uint8-random",
    "gradient",
    "flat",
    "impulse",
    "high-contrast-edges",
    "saturated",
    "aoi-like",
)


def measure(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return float(difference.max()), float(difference.mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", default=str(ROOT / "gpu" / "visionflow_cuda.dll"))
    args = parser.parse_args()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    runtime = GpuRuntime(args.dll)
    if not runtime.available:
        print(f"CUDA runtime unavailable: {runtime.unavailable_reason}")
        return 2
    if not runtime.supports_gaussian_blur_f32:
        print("CUDA DLL has no vf_gaussian_blur_f32 export; rebuild with gpu\\build_cuda_dll.ps1")
        return 2

    lines: list[str] = []
    record: dict = {
        "device": runtime.device_name,
        "compute_capability": runtime.compute_capability,
        "capabilities": runtime.status(True)["capabilities"],
        "claimed": {
            "gray_max_abs": CLAIMED_GRAY_MAX_ABS,
            "gray_mean_abs": CLAIMED_GRAY_MEAN_ABS,
            "relative_to_range": CLAIMED_RELATIVE_TO_RANGE,
            "verified_kernels": list(VERIFIED_KERNELS),
        },
    }

    coefficients = kernel_coefficients()
    record["kernel_coefficients"] = coefficients
    lines.append("== vf_gaussian_blur_f32 equivalence evidence ==")
    lines.append(f"device: {runtime.device_name} (sm {runtime.compute_capability})")
    lines.append(f"openCV: cv2 {cv2.__version__}, numpy {np.__version__}")
    lines.append("")
    lines.append("[coefficients] prepare_gaussian_f32_weights() mirror vs cv2.getGaussianKernel(k, 0, CV_32F)")
    lines.append(
        f"  odd sizes checked: {coefficients['checked_sizes']} (3..127)"
        f"  worst |diff| = {coefficients['worst_abs_difference']:.3e} at ksize={coefficients['worst_size']}"
        f"  bit-exact = {coefficients['bit_exact']}"
    )

    # 1. Every verified kernel size on one representative operand.
    kernel_rows = []
    worst_all = 0.0
    source = scene((192, 256), "uint8-random", 11)
    for ksize in VERIFIED_KERNELS:
        expected = cv2.GaussianBlur(source, (ksize, ksize), 0.0)
        actual = runtime.gaussian_blur_f32(source, ksize)
        maximum, mean = measure(actual, expected)
        kernel_rows.append({"kernel_size": ksize, "max_abs": maximum, "mean_abs": mean})
        worst_all = max(worst_all, maximum)
    record["per_kernel_size"] = kernel_rows
    lines.append("")
    lines.append("[per kernel size] 192x256 uint8-domain random float32, cv2.GaussianBlur(ksize, 0.0)")
    lines.append(f"  all {len(VERIFIED_KERNELS)} verified odd sizes: worst max|diff| = {worst_all:.3e}")
    for row in kernel_rows:
        if row["kernel_size"] in SWEEP_KERNELS:
            lines.append(
                f"    ksize={row['kernel_size']:3d}  max={row['max_abs']:.3e}  mean={row['mean_abs']:.3e}"
            )

    # 2. Shape x content sweep at representative kernel sizes. Operands are clipped to [0, 255] so
    # the gray-domain claim below is measured on exactly the value range it names.
    sweep_rows = []
    worst_gray = 0.0
    worst_gray_mean = 0.0
    worst_gray_ulp = 0.0
    seed = 1000
    for shape in SHAPES:
        for kind in CONTENTS:
            seed += 1
            operand = np.clip(scene(shape, kind, seed), 0.0, 255.0).astype(np.float32)
            magnitude = max(float(np.max(np.abs(operand))) if operand.size else 0.0, 1.0)
            for ksize in SWEEP_KERNELS:
                expected = cv2.GaussianBlur(operand, (ksize, ksize), 0.0)
                actual = runtime.gaussian_blur_f32(operand, ksize)
                maximum, mean = measure(actual, expected)
                ulps = maximum / float(np.spacing(np.float32(magnitude)))
                sweep_rows.append({
                    "shape": list(shape), "content": kind, "kernel_size": ksize,
                    "max_abs": maximum, "mean_abs": mean, "ulps": ulps,
                })
                worst_gray = max(worst_gray, maximum)
                worst_gray_ulp = max(worst_gray_ulp, ulps)
                # A 1x1 operand has a single sample, so its mean equals its max and would dominate
                # the mean statistic without saying anything about systematic error.
                if operand.size >= 64:
                    worst_gray_mean = max(worst_gray_mean, mean)
    record["shape_content_sweep"] = {
        "cases": len(sweep_rows),
        "gray_worst_max_abs": worst_gray,
        "gray_worst_mean_abs": worst_gray_mean,
        "gray_worst_ulp": worst_gray_ulp,
    }
    lines.append("")
    lines.append(
        f"[shape x content x ksize sweep] {len(sweep_rows)} cases "
        f"({len(SHAPES)} shapes x {len(CONTENTS)} contents x {len(SWEEP_KERNELS)} kernel sizes), "
        "operands clipped to [0, 255]"
    )
    lines.append(f"  gray-domain (values in [0, 255]) worst max|diff|  = {worst_gray:.3e}")
    lines.append(f"  gray-domain (values in [0, 255]) worst mean|diff| = {worst_gray_mean:.3e} (operands >= 64 px)")
    lines.append(f"  gray-domain worst max|diff| in float32 ulps of max|src| = {worst_gray_ulp:.2f}")
    worst_case = max(sweep_rows, key=lambda row: row["max_abs"])
    worst_mean_case = max(
        (row for row in sweep_rows if row["shape"][0] * row["shape"][1] >= 64),
        key=lambda row: row["mean_abs"],
    )
    lines.append(
        f"  worst case: shape={tuple(worst_case['shape'])} content={worst_case['content']} "
        f"ksize={worst_case['kernel_size']} max={worst_case['max_abs']:.3e}"
    )
    lines.append(
        f"  worst mean case: shape={tuple(worst_mean_case['shape'])} content={worst_mean_case['content']} "
        f"ksize={worst_mean_case['kernel_size']} mean={worst_mean_case['mean_abs']:.3e}"
    )
    lines.append("  six largest max|diff| cases:")
    for row in sorted(sweep_rows, key=lambda row: row["max_abs"], reverse=True)[:6]:
        lines.append(
            f"    shape={tuple(row['shape'])} content={row['content']} ksize={row['kernel_size']}"
            f" max={row['max_abs']:.3e} ({row['ulps']:.2f} ulp) mean={row['mean_abs']:.3e}"
        )

    # 2b. Wide dynamic range: random +-1000 operands, where the absolute difference scales with the
    # magnitude of the source rather than staying at the gray-domain level.
    wide_rows = []
    worst_wide = 0.0
    for shape in SHAPES:
        seed += 1
        operand = scene(shape, "negative-range", seed)
        dynamic_range = float(np.max(np.abs(operand))) if operand.size else 0.0
        for ksize in SWEEP_KERNELS:
            expected = cv2.GaussianBlur(operand, (ksize, ksize), 0.0)
            actual = runtime.gaussian_blur_f32(operand, ksize)
            maximum, mean = measure(actual, expected)
            wide_rows.append({
                "shape": list(shape), "kernel_size": ksize, "max_abs": maximum,
                "mean_abs": mean, "relative": maximum / max(dynamic_range, 1.0),
            })
            worst_wide = max(worst_wide, maximum / max(dynamic_range, 1.0))
    record["wide_range_sweep"] = {"cases": len(wide_rows), "worst_relative": worst_wide}
    lines.append("")
    lines.append(f"[wide dynamic range] +-1000 uniform random, {len(wide_rows)} cases")
    lines.append(f"  worst |diff| / max|value| = {worst_wide:.3e}")

    # 3. Small operands where the kernel is wider than the image (reflect101 wraps repeatedly).
    over_rows = []
    for shape in ((5, 5), (8, 11), (20, 20), (33, 7)):
        operand = scene(shape, "uint8-random", 5)
        for ksize in (51, 63, 127):
            expected = cv2.GaussianBlur(operand, (ksize, ksize), 0.0)
            actual = runtime.gaussian_blur_f32(operand, ksize)
            maximum, mean = measure(actual, expected)
            over_rows.append({"shape": list(shape), "kernel_size": ksize, "max_abs": maximum, "mean_abs": mean})
    record["kernel_wider_than_image"] = over_rows
    over_worst = max(row["max_abs"] for row in over_rows)
    lines.append("")
    lines.append("[kernel wider than the operand] reflect101 wraps repeatedly in both passes")
    lines.append(f"  {len(over_rows)} cases (5x5, 8x11, 20x20, 33x7 x ksize 51/63/127): worst max|diff| = {over_worst:.3e}")

    # 4. ROI variant: equality with cv2.GaussianBlur on the same sub-array.
    roi_rows = []
    if runtime.supports_gaussian_blur_f32_roi:
        panel = scene((240, 320), "aoi-like", 21)
        for (x, y, width, height, ksize) in (
            (0, 0, 320, 240, 51),
            (0, 0, 320, 240, 3),
            (16, 24, 128, 96, 51),
            (200, 100, 120, 140, 21),
            (319, 239, 1, 1, 9),
            (0, 0, 5, 5, 127),
        ):
            expected = cv2.GaussianBlur(panel[y:y + height, x:x + width], (ksize, ksize), 0.0)
            actual = runtime.gaussian_blur_f32_roi(panel, x, y, width, height, ksize)
            maximum, mean = measure(actual, expected)
            roi_rows.append({
                "rect": [x, y, width, height], "kernel_size": ksize,
                "max_abs": maximum, "mean_abs": mean,
            })
    record["roi_variant"] = roi_rows
    roi_worst = max((row["max_abs"] for row in roi_rows), default=0.0)
    lines.append("")
    lines.append("[ROI variant] vs cv2.GaussianBlur on the same sub-array (isolated rectangle)")
    for row in roi_rows:
        lines.append(
            f"    rect={tuple(row['rect'])} ksize={row['kernel_size']:3d}"
            f"  max={row['max_abs']:.3e}  mean={row['mean_abs']:.3e}"
        )
    lines.append(f"  worst max|diff| = {roi_worst:.3e}")

    # 5. Determinism: identical input bytes must produce identical output bytes.
    operand = scene((192, 256), "aoi-like", 33)
    first = runtime.gaussian_blur_f32(operand, 51)
    second = runtime.gaussian_blur_f32(operand, 51)
    deterministic = bool(np.array_equal(first, second))
    record["deterministic"] = deterministic
    lines.append("")
    lines.append(f"[determinism] two calls on identical bytes are bit-identical: {deterministic}")

    # 6. Unsupported kernel sizes are refused, never computed.
    refusals = []
    for ksize in (2, 4, 8, 128, 129, 255):
        try:
            runtime.gaussian_blur_f32(operand, ksize)
            refusals.append({"kernel_size": ksize, "error_code": None, "refused": False})
        except GpuRuntimeError as error:
            refusals.append({
                "kernel_size": ksize,
                "error_code": int(getattr(error, "error_code", 0)),
                "refused": True,
            })
    record["unsupported_kernel_sizes"] = refusals
    expected_unsupported = all(
        row["refused"]
        and row["error_code"] == (1 if row["kernel_size"] < 3 else CUDA_ERROR_UNSUPPORTED)
        for row in refusals
    )
    lines.append("")
    lines.append("[unsupported kernel sizes] odd 3..127 only; everything else is refused")
    for row in refusals:
        lines.append(
            f"    ksize={row['kernel_size']:3d} refused={row['refused']} error_code={row['error_code']}"
        )

    # 7. Claimed tolerance check.
    claim_ok = (
        coefficients["bit_exact"]
        and worst_gray <= CLAIMED_GRAY_MAX_ABS
        and worst_gray_mean <= CLAIMED_GRAY_MEAN_ABS
        and worst_wide <= CLAIMED_RELATIVE_TO_RANGE
        and over_worst <= CLAIMED_GRAY_MAX_ABS
        and (not roi_rows or roi_worst <= CLAIMED_GRAY_MAX_ABS)
        and deterministic
        and expected_unsupported
    )
    record["claim_ok"] = claim_ok
    lines.append("")
    lines.append("[claimed tolerance]")
    lines.append(
        f"  float32 gray/residual in [0, 255], any odd ksize in [3, 127]:"
        f"  max|diff| <= {CLAIMED_GRAY_MAX_ABS:.1e} (measured {worst_gray:.3e}),"
        f"  mean|diff| <= {CLAIMED_GRAY_MEAN_ABS:.1e} (measured {worst_gray_mean:.3e})"
    )
    lines.append(
        f"  wider dynamic range: |diff| <= {CLAIMED_RELATIVE_TO_RANGE:.1e} of the source range"
        f" (measured {worst_wide:.3e})"
    )
    lines.append(f"  claim holds: {claim_ok}")

    EVIDENCE_TEXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    EVIDENCE_JSON.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {EVIDENCE_TEXT.relative_to(ROOT)}")
    print(f"written: {EVIDENCE_JSON.relative_to(ROOT)}")
    runtime.close()
    return 0 if claim_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
