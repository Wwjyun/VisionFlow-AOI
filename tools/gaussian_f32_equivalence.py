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
# Explicit (non-zero) sigma takes the pure exp/sum/normalize branch for every ksize, where OpenCV's
# SIMD filter diverges a little more; measured separately and claimed separately.
CLAIMED_SIGMA_MAX_ABS = 4.0e-4
CLAIMED_SIGMA_MEAN_ABS = 5.0e-5
# The error scales with the magnitude of the source. For wide-range inputs the claim is relative to
# the dynamic range instead of an absolute float32 unit.
CLAIMED_RELATIVE_TO_RANGE = 5.0e-7

VERIFIED_KERNELS = tuple(range(3, 128, 2))

# Fixed representative kernel sizes used for the shape/content sweep.
SWEEP_KERNELS = (3, 5, 9, 21, 51, 63, 101, 127)
# Sigma values the sweep covers. Zero and negative both select OpenCV's automatic sigma rule.
SWEEP_SIGMAS = (0.0, 0.5, 1.0, 1.25, 2.5, 5.0)
SIGMA_SWEEP_KERNELS = (3, 5, 9, 21, 31, 51, 127)
SIGMA_SWEEP_SHAPES = ((64, 80), (400, 600), (3, 1024))
# Coefficients are checked for every kernel size against every sigma, including a negative sigma
# (auto alias) and a sigma equal to the automatic value of that kernel size.
KERNEL_COEFFICIENT_SIGMAS = (0.0, 0.5, 1.0, 1.25, 2.5, 5.0, -1.0)
KERNEL_COEFFICIENT_SIZES = VERIFIED_KERNELS

# OpenCV's fixed small-kernel table (SMALL_GAUSSIAN_SIZE == 9 in OpenCV 5.x), indexed by ksize / 2.
SMALL_KERNELS = {
    1: (1.0,),
    3: (0.25, 0.5, 0.25),
    5: (0.0625, 0.25, 0.375, 0.25, 0.0625),
    7: (0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125),
    9: (0.015625, 0.05078125, 0.1171875, 0.19921875, 0.234375,
        0.19921875, 0.1171875, 0.05078125, 0.015625),
}


def automatic_sigma(ksize: int) -> float:
    return ((ksize - 1) * 0.5 - 1) * 0.3 + 0.8


def reference_kernel_f32(ksize: int, sigma: float = 0.0) -> np.ndarray:
    """Mirror of prepare_gaussian_f32_weights() in gpu/visionflow_cuda.cu.

    ``sigma > 0`` is used directly; zero or negative selects OpenCV's automatic rule, including the
    fixed small-kernel table for odd ksize <= 9.
    """
    automatic = not sigma > 0.0
    if automatic and ksize <= 9:
        return np.array(SMALL_KERNELS[ksize], dtype=np.float32)
    sigma_x = automatic_sigma(ksize) if automatic else float(sigma)
    offset = np.arange(ksize, dtype=np.float64) - (ksize - 1) * 0.5
    raw = np.exp(-0.5 * offset * offset / (sigma_x * sigma_x))
    return (raw * (1.0 / float(np.sum(raw)))).astype(np.float32)


def kernel_coefficients() -> dict:
    """Check the generated coefficients against cv2.getGaussianKernel for every verified size.

    Every combination of kernel size and sigma in the sweep must be bit-identical; a non-zero sigma
    must NOT fall back to the automatic table, and a non-positive sigma must select it.
    """
    worst = 0.0
    worst_case = None
    checked = 0
    for ksize in KERNEL_COEFFICIENT_SIZES:
        for sigma in KERNEL_COEFFICIENT_SIGMAS:
            reference = cv2.getGaussianKernel(ksize, sigma, cv2.CV_32F).reshape(-1)
            difference = float(np.max(np.abs(reference - reference_kernel_f32(ksize, sigma))))
            checked += 1
            if difference > worst:
                worst, worst_case = difference, (ksize, sigma)
    # The failure mode the coefficient rule exists to prevent: a non-zero sigma must not silently
    # produce the automatic-sigma kernel.
    distinct = True
    for ksize in KERNEL_COEFFICIENT_SIZES:
        for sigma in (0.5, 1.0, 2.5):
            if np.array_equal(
                reference_kernel_f32(ksize, sigma), reference_kernel_f32(ksize, 0.0)
            ):
                distinct = False
    return {
        "checked_combinations": checked,
        "worst_abs_difference": worst,
        "worst_case": list(worst_case) if worst_case else None,
        "bit_exact": worst == 0.0,
        "non_zero_sigma_differs_from_auto": distinct,
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
    lines.append("[coefficients] prepare_gaussian_f32_weights() mirror vs cv2.getGaussianKernel(k, sigma, CV_32F)")
    lines.append(
        f"  combinations checked: {coefficients['checked_combinations']} "
        f"(odd ksize 3..127 x sigma {KERNEL_COEFFICIENT_SIGMAS})"
    )
    lines.append(
        f"  worst |diff| = {coefficients['worst_abs_difference']:.3e} at "
        f"(ksize, sigma)={tuple(coefficients['worst_case']) if coefficients['worst_case'] else None}"
        f"  bit-exact = {coefficients['bit_exact']}"
    )
    lines.append(
        "  every non-zero sigma produces a kernel different from the automatic-sigma kernel: "
        f"{coefficients['non_zero_sigma_differs_from_auto']}"
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

    # 1b. Non-zero sigma: the export must use that sigma, not OpenCV's automatic rule. This is the
    # regression the sigma=0-only sweep could not see.
    sigma_rows = []
    worst_sigma = 0.0
    worst_sigma_mean = 0.0
    worst_sigma_case = None
    seed = 500
    for shape in SIGMA_SWEEP_SHAPES:
        for kind in CONTENTS:
            seed += 1
            operand = np.clip(scene(shape, kind, seed), 0.0, 255.0).astype(np.float32)
            for ksize in SIGMA_SWEEP_KERNELS:
                for sigma in SWEEP_SIGMAS:
                    expected = cv2.GaussianBlur(operand, (ksize, ksize), sigma)
                    actual = runtime.gaussian_blur_f32(operand, ksize, sigma)
                    maximum, mean = measure(actual, expected)
                    # Distance from the automatic-sigma result: a device that ignored sigma would
                    # match cv2(sigma=0) here instead of cv2(sigma).
                    auto_reference = cv2.GaussianBlur(operand, (ksize, ksize), 0.0)
                    auto_gap = float(np.max(np.abs(actual.astype(np.float64) - auto_reference.astype(np.float64))))
                    sigma_rows.append({
                        "shape": list(shape), "content": kind, "kernel_size": ksize, "sigma": sigma,
                        "max_abs": maximum, "mean_abs": mean, "gap_to_auto_sigma": auto_gap,
                    })
                    if maximum > worst_sigma:
                        worst_sigma, worst_sigma_case = maximum, (shape, kind, ksize, sigma)
                    worst_sigma_mean = max(worst_sigma_mean, mean)
    record["sigma_sweep"] = {
        "cases": len(sigma_rows),
        "worst_max_abs": worst_sigma,
        "worst_mean_abs": worst_sigma_mean,
        "worst_case": [list(worst_sigma_case[0]), worst_sigma_case[1], worst_sigma_case[2],
                       worst_sigma_case[3]] if worst_sigma_case else None,
        "sigmas": list(SWEEP_SIGMAS),
        "kernel_sizes": list(SIGMA_SWEEP_KERNELS),
    }
    lines.append("")
    lines.append(
        f"[sigma sweep] {len(sigma_rows)} cases "
        f"({len(SIGMA_SWEEP_SHAPES)} shapes x {len(CONTENTS)} contents x "
        f"{len(SIGMA_SWEEP_KERNELS)} kernel sizes x {len(SWEEP_SIGMAS)} sigma values)"
    )
    lines.append(
        f"  sigma values {SWEEP_SIGMAS}, kernel sizes {SIGMA_SWEEP_KERNELS}"
    )
    lines.append(f"  worst max|device - cv2(sigma)|  = {worst_sigma:.3e}")
    lines.append(f"  worst mean|device - cv2(sigma)| = {worst_sigma_mean:.3e}")
    if worst_sigma_case:
        lines.append(
            f"  worst case: shape={worst_sigma_case[0]} content={worst_sigma_case[1]} "
            f"ksize={worst_sigma_case[2]} sigma={worst_sigma_case[3]}"
        )
    non_auto_rows = [row for row in sigma_rows if row["sigma"] > 0.0]
    for sigma in SWEEP_SIGMAS:
        if sigma <= 0.0:
            continue
        rows = [row for row in non_auto_rows if row["sigma"] == sigma]
        lines.append(
            f"  sigma={sigma:<5} cases={len(rows):4d} worst max|diff| = "
            f"{max(row['max_abs'] for row in rows):.3e}  "
            f"max|device - cv2(auto sigma)| = {max(row['gap_to_auto_sigma'] for row in rows):.3e}"
        )

    # 1c. A non-positive sigma is OpenCV's automatic rule, so it must reproduce cv2(sigma=0).
    auto_alias_rows = []
    for ksize in (3, 9, 31, 51):
        operand = np.clip(scene((400, 600), "uint8-random", 900 + ksize), 0.0, 255.0).astype(np.float32)
        for sigma in (0.0, -1.0, -0.25):
            expected = cv2.GaussianBlur(operand, (ksize, ksize), 0.0)
            actual = runtime.gaussian_blur_f32(operand, ksize, sigma)
            maximum, mean = measure(actual, expected)
            auto_alias_rows.append({
                "kernel_size": ksize, "sigma": sigma, "max_abs": maximum, "mean_abs": mean,
            })
    record["auto_sigma_alias"] = auto_alias_rows
    lines.append("")
    lines.append("[non-positive sigma == OpenCV automatic rule] vs cv2.GaussianBlur(ksize, 0.0)")
    for row in auto_alias_rows:
        lines.append(
            f"    ksize={row['kernel_size']:3d} sigma={row['sigma']:>5} "
            f"max={row['max_abs']:.3e} mean={row['mean_abs']:.3e}"
        )

    # 1d. A sigma with no OpenCV coefficient rule must be refused, never folded into the auto rule.
    sigma_refusals = []
    for sigma in (float("nan"), float("inf"), float("-inf")):
        try:
            runtime.gaussian_blur_f32(source, 31, sigma)
            sigma_refusals.append({"sigma": repr(sigma), "refused": False, "error_code": None})
        except GpuRuntimeError as error:
            sigma_refusals.append({
                "sigma": repr(sigma), "refused": True,
                "error_code": int(getattr(error, "error_code", 0)),
            })
    record["sigma_refusals"] = sigma_refusals
    lines.append("")
    lines.append("[non-finite sigma] refused with VF_CUDA_INVALID_ARGUMENT (1)")
    for row in sigma_refusals:
        lines.append(f"    sigma={row['sigma']:>5} refused={row['refused']} error_code={row['error_code']}")

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

    # 4. ROI variant: equality with cv2.GaussianBlur on the same sub-array, for auto and explicit
    # sigma.
    roi_rows = []
    if runtime.supports_gaussian_blur_f32_roi:
        panel = scene((240, 320), "aoi-like", 21)
        for (x, y, width, height, ksize, sigma) in (
            (0, 0, 320, 240, 51, 0.0),
            (0, 0, 320, 240, 3, 0.0),
            (16, 24, 128, 96, 51, 0.0),
            (200, 100, 120, 140, 21, 0.0),
            (319, 239, 1, 1, 9, 0.0),
            (0, 0, 5, 5, 127, 0.0),
            (16, 24, 128, 96, 51, 1.25),
            (100, 60, 64, 64, 31, 2.5),
            (0, 0, 320, 240, 9, 0.5),
        ):
            expected = cv2.GaussianBlur(panel[y:y + height, x:x + width], (ksize, ksize), sigma)
            actual = runtime.gaussian_blur_f32_roi(panel, x, y, width, height, ksize, sigma)
            maximum, mean = measure(actual, expected)
            roi_rows.append({
                "rect": [x, y, width, height], "kernel_size": ksize, "sigma": sigma,
                "max_abs": maximum, "mean_abs": mean,
            })
    record["roi_variant"] = roi_rows
    roi_worst = max((row["max_abs"] for row in roi_rows), default=0.0)
    lines.append("")
    lines.append("[ROI variant] vs cv2.GaussianBlur on the same sub-array (isolated rectangle)")
    for row in roi_rows:
        lines.append(
            f"    rect={tuple(row['rect'])} ksize={row['kernel_size']:3d} sigma={row['sigma']:<5}"
            f"  max={row['max_abs']:.3e}  mean={row['mean_abs']:.3e}"
        )
    lines.append(f"  worst max|diff| = {roi_worst:.3e}")

    # 5. Determinism: identical input bytes must produce identical output bytes.
    operand = scene((192, 256), "aoi-like", 33)
    first = runtime.gaussian_blur_f32(operand, 51)
    second = runtime.gaussian_blur_f32(operand, 51)
    deterministic = bool(np.array_equal(first, second))
    sigma_deterministic = bool(
        np.array_equal(
            runtime.gaussian_blur_f32(operand, 51, 1.25),
            runtime.gaussian_blur_f32(operand, 51, 1.25),
        )
    )
    record["deterministic"] = deterministic and sigma_deterministic
    record["deterministic_explicit_sigma"] = sigma_deterministic
    lines.append("")
    lines.append(
        f"[determinism] two calls on identical bytes are bit-identical: "
        f"auto sigma={deterministic}, explicit sigma=1.25 {sigma_deterministic}"
    )

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

    # 7. The bridge must refuse a DLL that ignores sigma rather than returning the automatic-sigma
    # background for it. The legacy behaviour is simulated by dropping the trailing argument.
    native_blur = runtime._dll.vf_gaussian_blur_f32
    legacy_rows: dict = {}
    try:
        runtime._dll.vf_gaussian_blur_f32 = lambda *values: native_blur(*values[:-1], 0.0)
        legacy_rows["probe_detects_legacy"] = runtime._probe_gaussian_f32_sigma() is False
        runtime._gaussian_f32_sigma_supported = False
        try:
            runtime.gaussian_blur_f32(source, 31, 1.0)
            legacy_rows["refuses_explicit_sigma"] = False
        except GpuRuntimeError as error:
            legacy_rows["refuses_explicit_sigma"] = True
            legacy_rows["refusal_message"] = str(error)
        try:
            runtime.gaussian_blur_f32(source, 31, 0.0)
            legacy_rows["still_allows_automatic_sigma"] = True
        except GpuRuntimeError:
            legacy_rows["still_allows_automatic_sigma"] = False
    finally:
        runtime._dll.vf_gaussian_blur_f32 = native_blur
        runtime._gaussian_f32_sigma_supported = runtime._probe_gaussian_f32_sigma()
    legacy_rows["probe_accepts_current_dll"] = bool(runtime.supports_gaussian_f32_sigma)
    record["sigma_support_guard"] = legacy_rows
    lines.append("")
    lines.append("[sigma support guard] a DLL whose export ignores sigma must be refused")
    lines.append(f"  probe detects the simulated legacy export: {legacy_rows.get('probe_detects_legacy')}")
    lines.append(f"  explicit sigma refused on such a DLL: {legacy_rows.get('refuses_explicit_sigma')}")
    lines.append(
        f"  automatic sigma still allowed on such a DLL: "
        f"{legacy_rows.get('still_allows_automatic_sigma')}"
    )
    lines.append(f"  probe accepts this DLL: {legacy_rows['probe_accepts_current_dll']}")
    if legacy_rows.get("refusal_message"):
        lines.append(f"  refusal: {legacy_rows['refusal_message']}")

    # 8. Claimed tolerance check. Ignoring sigma (the regression this sweep exists for) shows up as
    # a huge max_abs on every sigma > 0 case, so the sigma bound below is the real gate.
    sigma_refusals_ok = all(row["refused"] and row["error_code"] == 1 for row in sigma_refusals)
    auto_alias_ok = all(row["max_abs"] <= CLAIMED_GRAY_MAX_ABS for row in auto_alias_rows)
    guard_ok = (
        legacy_rows.get("probe_detects_legacy")
        and legacy_rows.get("refuses_explicit_sigma")
        and legacy_rows.get("still_allows_automatic_sigma")
        and legacy_rows.get("probe_accepts_current_dll")
    )
    claim_ok = (
        coefficients["bit_exact"]
        and coefficients["non_zero_sigma_differs_from_auto"]
        and worst_gray <= CLAIMED_GRAY_MAX_ABS
        and worst_gray_mean <= CLAIMED_GRAY_MEAN_ABS
        and worst_wide <= CLAIMED_RELATIVE_TO_RANGE
        and over_worst <= CLAIMED_GRAY_MAX_ABS
        and (not roi_rows or roi_worst <= CLAIMED_GRAY_MAX_ABS)
        and deterministic
        and sigma_deterministic
        and expected_unsupported
        and sigma_refusals_ok
        and auto_alias_ok
        and guard_ok
        and worst_sigma <= CLAIMED_SIGMA_MAX_ABS
        and worst_sigma_mean <= CLAIMED_SIGMA_MEAN_ABS
    )
    record["claim_ok"] = claim_ok
    lines.append("")
    lines.append("[claimed tolerance]")
    lines.append(
        f"  float32 gray/residual in [0, 255], any odd ksize in [3, 127], sigma <= 0 (automatic):"
        f"  max|diff| <= {CLAIMED_GRAY_MAX_ABS:.1e} (measured {worst_gray:.3e}),"
        f"  mean|diff| <= {CLAIMED_GRAY_MEAN_ABS:.1e} (measured {worst_gray_mean:.3e})"
    )
    lines.append(
        f"  float32 gray/residual in [0, 255], any odd ksize in [3, 127], sigma > 0 (explicit):"
        f"  max|diff| <= {CLAIMED_SIGMA_MAX_ABS:.1e} (measured {worst_sigma:.3e}),"
        f"  mean|diff| <= {CLAIMED_SIGMA_MEAN_ABS:.1e} (measured {worst_sigma_mean:.3e})"
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
