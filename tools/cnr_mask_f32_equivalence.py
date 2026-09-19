"""Acceptance gate for the optional CNR mask export (vf_cnr_mask_f32).

Compares the CUDA export against the reference the 202-CS-SN-1 detector uses in
``detectors/detector_202_1.py::_automatic_cnr_mask``: the float32 Gaussian background via
``cv2.GaussianBlur`` on the host, then

    residual        = image_float - background
    residual_median = np.median(residual)
    mad             = np.median(np.abs(residual - residual_median))
    robust          = max(mad_scale * mad, noise_sigma_floor)
    threshold       = max(residual_threshold_floor, residual_sigma_multiplier * robust)
    mask            = (np.abs(residual - residual_median) > threshold).astype(np.uint8) * max_value

The two medians must be *bit-identical* to ``np.median`` on float32 (including the even-count
float32 average and the NaN case), the threshold must be identical as a double, and the mask must be
byte-identical. No tolerance is claimed: a single differing bit or byte fails the gate. Run with the
repository Python:

    .\\env\\Scripts\\python.exe tools\\cnr_mask_f32_equivalence.py

Evidence is written to outputs_validation/cnr_profile/cnr_mask_f32_equivalence.txt (human summary)
and .json (machine readable). The tool exits non-zero when any comparison fails, when a refusal case
is accepted, or when the runtime has no vf_cnr_mask_f32 export.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gpu_runtime import (  # noqa: E402
    GpuRuntime,
)

EVIDENCE_DIR = ROOT / "outputs_validation" / "cnr_profile"
EVIDENCE_TEXT = EVIDENCE_DIR / "cnr_mask_f32_equivalence.txt"
EVIDENCE_JSON = EVIDENCE_DIR / "cnr_mask_f32_equivalence.json"

# detectors/detector_202_1.py defaults: residual_sigma_multiplier, residual_threshold_floor,
# noise_sigma_floor, mad_scale and candidate_max_value.
SIGMA_MULTIPLIER = 3.0
THRESHOLD_FLOOR = 8.0
ABSOLUTE_FLOOR = 1e-6
MAD_SCALE = 1.4826
CANDIDATE_VALUE = 255

# VF_CUDA_INVALID_ARGUMENT: every malformed request in this tool must be refused with it.
CUDA_ERROR_INVALID_ARGUMENT = 1

KERNELS = (3, 9, 31, 51, 127)
SIGMAS = (0.0, 1.0, 2.5)
# Full cross product for the cheap shapes.
SWEEP = tuple((kernel, sigma) for sigma in SIGMAS for kernel in KERNELS)
# The two production ROI shapes keep every kernel at the production sigma 0.0 plus the 51 kernel at
# the two non-zero sigmas, because each case costs a 24-megapixel cv2 filter and two NumPy medians.
PRODUCTION_SWEEP = (
    (3, 0.0), (9, 0.0), (31, 0.0), (51, 0.0), (127, 0.0), (51, 1.0), (51, 2.5),
)
PRODUCTION_SIZES = ((2000, 4000), (2000, 12000))
SIZES = ((1, 1), (1, 17), (64, 80), (400, 600), (1024, 192)) + PRODUCTION_SIZES
# The scene kinds the task requires: constant, gradient, impulse, high-contrast edges, uniform
# random, one NaN and all-equal content.
KINDS = (
    "constant",
    "all-equal",
    "gradient",
    "impulse",
    "high-contrast-edges",
    "uniform-random",
    "single-nan",
)

TIMING_SHAPE = (2000, 12000)
TIMING_KERNEL = 51
TIMING_SIGMA = 0.0
TIMING_REPEATS = 3

# Argument order of vf_cnr_mask_f32; raw calls must be positional because a ctypes function pointer
# does not accept keyword arguments.
ARG_ORDER = (
    "context",
    "image", "image_stride",
    "background", "background_stride",
    "width", "height",
    "sigma_multiplier", "threshold_floor", "absolute_floor", "mad_scale",
    "candidate_value",
    "out_residual_median", "out_mad", "out_threshold",
    "out_mask", "out_mask_capacity",
)


def make_scene(shape: tuple[int, int], kind: str, seed: int) -> np.ndarray:
    """Build one deterministic float32 test image of the requested shape and kind."""
    height, width = shape
    rng = np.random.default_rng(seed)
    if kind == "constant":
        return (np.full(shape, 128.0, dtype=np.float32)
                + rng.uniform(-0.5, 0.5, shape).astype(np.float32)).astype(np.float32)
    if kind == "all-equal":
        return np.full(shape, 42.0, dtype=np.float32)
    if kind == "gradient":
        y, x = np.mgrid[0:height, 0:width]
        ramp = (x * (255.0 / max(width - 1, 1)) + y * (64.0 / max(height - 1, 1)))
        return ramp.astype(np.float32)
    if kind == "impulse":
        data = np.full(shape, 100.0, dtype=np.float32)
        data[height // 2, width // 2] = 255.0
        data[height // 3, width // 3] = 0.0
        if height > 2 and width > 2:
            data[1, 1] = 255.0
        return data
    if kind == "high-contrast-edges":
        data = np.zeros(shape, dtype=np.float32)
        data[:, width // 2:] = 255.0
        data[height // 3: 2 * height // 3, width // 4:] = 0.0
        data[height // 5, :] = 255.0
        return data
    if kind == "uniform-random":
        return rng.uniform(0.0, 255.0, shape).astype(np.float32)
    if kind == "single-nan":
        data = (np.full(shape, 150.0, dtype=np.float32)
                + rng.normal(0.0, 2.0, shape).astype(np.float32)).astype(np.float32)
        data[height // 2, width // 2] = np.float32("nan")
        return data
    raise ValueError(f"unknown scene kind: {kind}")


def reference_cnr_steps(
    image: np.ndarray,
    background: np.ndarray,
    *,
    sigma_multiplier: float = SIGMA_MULTIPLIER,
    threshold_floor: float = THRESHOLD_FLOOR,
    absolute_floor: float = ABSOLUTE_FLOOR,
    mad_scale: float = MAD_SCALE,
    candidate_value: int = CANDIDATE_VALUE,
) -> tuple[np.float32, np.float32, float, np.ndarray]:
    """The steps of _automatic_cnr_mask after the Gaussian background exists, verbatim.

    ``mad_scale`` and the medians are Python floats here exactly as they are in the detector
    (``_exact_median`` returns ``float(...)``), so ``mad_scale * mad``, both ``max`` calls and
    ``sigma_multiplier * robust`` are double computations, and NumPy narrows the threshold to float32
    for the mask comparison.
    """
    residual = image - background
    residual_median = np.median(residual)
    mad = np.median(np.abs(residual - residual_median))
    robust_noise_sigma = max(float(mad_scale) * float(mad), float(absolute_floor))
    threshold = max(float(threshold_floor), float(sigma_multiplier) * robust_noise_sigma)
    mask = (np.abs(residual - residual_median) > threshold).astype(np.uint8) * int(candidate_value)
    return residual_median, mad, threshold, mask


def scalar_state(device: np.float32, reference: np.float32) -> str:
    """Compare two float32 results: exact bits, both NaN, or a mismatch."""
    device_nan = bool(np.isnan(device))
    reference_nan = bool(np.isnan(reference))
    if device_nan and reference_nan:
        return "nan"
    if device_nan or reference_nan:
        return "mismatch"
    if np.float32(device).tobytes() == np.float32(reference).tobytes():
        return "exact"
    return "mismatch"


def device_call(runtime: GpuRuntime, image, background, **overrides):
    """Call the bridge and unpack the documented mapping into a positional tuple."""
    parameters = {
        "sigma_multiplier": SIGMA_MULTIPLIER,
        "threshold_floor": THRESHOLD_FLOOR,
        "absolute_floor": ABSOLUTE_FLOOR,
        "mad_scale": MAD_SCALE,
        "candidate_value": CANDIDATE_VALUE,
    }
    parameters.update(overrides)
    result = runtime.cnr_mask_f32(image, background, **parameters)
    return (
        result["residual_median"],
        result["mad"],
        result["threshold"],
        result["mask"],
    )


def result_contract(runtime: GpuRuntime) -> dict:
    """The bridge must return the mapping the detector's call site reads.

    ``detectors/detector_202_1.py`` reads ``device["mask"]``, ``device["residual_median"]``,
    ``device["mad"]`` and ``device["threshold"]``; a return shape that only looks similar would be
    swallowed by that call site's ``except Exception`` and silently disable the device path, so the
    keys are pinned here.
    """
    image = np.full((16, 24), 120.0, dtype=np.float32)
    image[4:6, 6:10] = 250.0
    background = cv2.GaussianBlur(image, (9, 9), 0.0)
    result = runtime.cnr_mask_f32(
        image, background,
        sigma_multiplier=SIGMA_MULTIPLIER, threshold_floor=THRESHOLD_FLOOR,
        absolute_floor=ABSOLUTE_FLOOR, mad_scale=MAD_SCALE, candidate_value=CANDIDATE_VALUE,
    )
    required = ("residual_median", "mad", "threshold", "mask")
    return {
        "type": type(result).__name__,
        "is_mapping": hasattr(result, "keys"),
        "keys": sorted(result.keys()) if hasattr(result, "keys") else [],
        "has_required_keys": all(key in result for key in required),
        "exact_key_set": sorted(result.keys()) == sorted(required) if hasattr(result, "keys") else False,
        "median_type": type(result["residual_median"]).__name__ if hasattr(result, "keys") else None,
        "mad_type": type(result["mad"]).__name__ if hasattr(result, "keys") else None,
        "threshold_type": type(result["threshold"]).__name__ if hasattr(result, "keys") else None,
        "mask_type": type(result["mask"]).__name__ if hasattr(result, "keys") else None,
        "mask_shape": list(result["mask"].shape) if hasattr(result, "keys") else None,
    }


def sweep_cases(runtime: GpuRuntime) -> dict:
    """Compare the export against the reference over every scene x (kernel, sigma) case."""
    summary = {
        "cases": 0,
        "median_exact": 0,
        "median_nan": 0,
        "median_mismatch": 0,
        "mad_exact": 0,
        "mad_nan": 0,
        "mad_mismatch": 0,
        "mask_identical": 0,
        "mask_mismatch": 0,
        "mask_differing_pixels": 0,
        "threshold_exact": 0,
        "threshold_mismatch": 0,
        "threshold_max_abs_diff": 0.0,
        "threshold_worst_case": None,
        "failures": [],
    }
    per_size: list[dict] = []
    seed = 202_000
    for shape in SIZES:
        pairs = PRODUCTION_SWEEP if shape in PRODUCTION_SIZES else SWEEP
        size_summary = {"shape": list(shape), "pairs": len(pairs), "cases": 0, "failures": 0}
        for kind in KINDS:
            seed += 1
            image = np.ascontiguousarray(make_scene(shape, kind, seed), dtype=np.float32)
            for kernel, sigma in pairs:
                background = cv2.GaussianBlur(image, (kernel, kernel), sigma)
                reference = reference_cnr_steps(image, background)
                device = device_call(runtime, image, background)
                case = {
                    "shape": list(shape), "kind": kind, "kernel": kernel, "sigma": sigma,
                }
                summary["cases"] += 1
                size_summary["cases"] += 1

                median_state = scalar_state(device[0], reference[0])
                mad_state = scalar_state(device[1], reference[1])
                summary[f"median_{median_state}"] += 1
                summary[f"mad_{mad_state}"] += 1

                threshold_diff = abs(float(device[2]) - float(reference[2]))
                if threshold_diff == 0.0:
                    summary["threshold_exact"] += 1
                else:
                    summary["threshold_mismatch"] += 1
                    if threshold_diff > summary["threshold_max_abs_diff"]:
                        summary["threshold_max_abs_diff"] = threshold_diff
                        summary["threshold_worst_case"] = dict(case)

                mask_equal = bool(np.array_equal(device[3], reference[3]))
                if mask_equal:
                    summary["mask_identical"] += 1
                else:
                    differing = int(np.count_nonzero(device[3] != reference[3]))
                    summary["mask_mismatch"] += 1
                    summary["mask_differing_pixels"] += differing
                    case["differing_pixels"] = differing

                if median_state == "mismatch" or mad_state == "mismatch" or threshold_diff != 0.0 \
                        or not mask_equal:
                    size_summary["failures"] += 1
                    if len(summary["failures"]) < 12:
                        summary["failures"].append({
                            **case,
                            "median_state": median_state,
                            "mad_state": mad_state,
                            "device_median": float(device[0]),
                            "reference_median": float(reference[0]),
                            "device_mad": float(device[1]),
                            "reference_mad": float(reference[1]),
                            "device_threshold": float(device[2]),
                            "reference_threshold": float(reference[2]),
                            "mask_equal": mask_equal,
                        })
        per_size.append(size_summary)
    summary["per_size"] = per_size
    return summary


def threshold_rounding_probe(runtime: GpuRuntime) -> dict:
    """Probe the float32 comparison semantics the NumPy reference uses for the mask.

    The reference compares ``absdev > residual_threshold`` where the threshold is a Python float, so
    NumPy narrows it to float32 before comparing. A device that compared the double instead would
    differ only when the double rounds *up* to its float32 neighbour and a pixel sits exactly on that
    float32 value. This probe constructs that case: ``background`` is all zeros, so the residual is
    the image itself, and the residual multiset is built so that ``np.median`` is exactly 0 and
    ``mad`` is exactly 2.0 - a magnitude whose threshold 3.0 * 1.4826 * 2.0 = 8.8956 narrows up to the
    float32 value 8.895600318908691. Pixels holding exactly that float32 value, one float32 step
    below it and one step above it are placed at the extremes of the distribution (the negatives in
    the negative half, the positives in the positive half) so both medians stay pinned.
    """
    height, width = 64, 80
    total = height * width
    half = total // 2
    values = np.empty(total, dtype=np.float32)
    values[:half] = -2.0
    values[half:] = 2.0
    threshold_double = max(
        THRESHOLD_FLOOR, SIGMA_MULTIPLIER * max(MAD_SCALE * 2.0, ABSOLUTE_FLOOR)
    )
    threshold_f32 = np.float32(threshold_double)
    above = np.float32(np.nextafter(threshold_f32, np.float32(np.inf)))
    below = np.float32(np.nextafter(threshold_f32, np.float32(-np.inf)))
    # One negative probe inside the negative half, four positive probes inside the positive half:
    # the residual multiset stays balanced, so median(residual) == 0 and median(|residual|) == 2.0.
    values[0] = -above
    values[half:half + 4] = np.array([below, threshold_f32, threshold_f32, above], dtype=np.float32)
    image = values.reshape(height, width).copy()
    background = np.zeros((height, width), dtype=np.float32)
    reference = reference_cnr_steps(image, background)
    device = device_call(runtime, image, background)

    residual = image - background
    absdev = np.abs(residual - reference[0])
    float32_style = absdev > float(reference[2])
    double_style = absdev.astype(np.float64) > float(reference[2])
    return {
        "premise_median_is_zero": bool(float(reference[0]) == 0.0),
        "premise_mad_is_two": bool(float(reference[1]) == 2.0),
        "threshold_double": float(threshold_double),
        "threshold_float32": float(threshold_f32),
        "threshold_rounds_up_to_float32": float(threshold_f32) > float(threshold_double),
        "probe_pixels": 5,
        "device_mask_ones": int(np.count_nonzero(device[3])),
        "python_mask_ones": int(np.count_nonzero(float32_style)),
        "double_comparison_mask_ones": int(np.count_nonzero(double_style)),
        "probe_is_discriminating": int(np.count_nonzero(float32_style))
        != int(np.count_nonzero(double_style)),
        "mask_identical": bool(np.array_equal(device[3], reference[3])),
        "median_exact": scalar_state(device[0], reference[0]) != "mismatch",
        "mad_exact": scalar_state(device[1], reference[1]) != "mismatch",
        "threshold_exact": abs(float(device[2]) - float(reference[2])) == 0.0,
    }


def stride_case(runtime: GpuRuntime) -> dict:
    """A rectangular ROI of a wider plane must be accepted by its byte stride, not copied."""
    height, width = 96, 128
    panel_width = 200
    rng = np.random.default_rng(4242)
    panel = rng.uniform(0.0, 255.0, (height, panel_width)).astype(np.float32)
    view = panel[:, 10: 10 + width]
    reference_image = np.ascontiguousarray(view)
    background_roi = cv2.GaussianBlur(reference_image, (31, 31), 0.0)
    reference = reference_cnr_steps(reference_image, background_roi)
    device = device_call(runtime, view, background_roi)
    return {
        "non_contiguous": bool(not view.flags["C_CONTIGUOUS"]),
        "view_strides": [int(view.strides[0]), int(view.strides[1])],
        "view_bytes_used": int(view.nbytes),
        "median_exact": scalar_state(device[0], reference[0]) != "mismatch",
        "mad_exact": scalar_state(device[1], reference[1]) != "mismatch",
        "threshold_exact": abs(float(device[2]) - float(reference[2])) == 0.0,
        "mask_identical": bool(np.array_equal(device[3], reference[3])),
    }


def determinism_and_non_mutation(runtime: GpuRuntime) -> dict:
    """Two identical calls must agree bit for bit and neither operand may be written."""
    shape = (240, 320)
    rng = np.random.default_rng(99)
    image = rng.uniform(0.0, 255.0, shape).astype(np.float32)
    image[50:60, 70:90] = 255.0
    background = cv2.GaussianBlur(image, (51, 51), 0.0)
    image_backup = image.copy()
    background_backup = background.copy()

    first = device_call(runtime, image, background)
    second = device_call(runtime, image, background)
    return {
        "median_bit_identical": np.float32(first[0]).tobytes() == np.float32(second[0]).tobytes(),
        "mad_bit_identical": np.float32(first[1]).tobytes() == np.float32(second[1]).tobytes(),
        "threshold_identical": float(first[2]) == float(second[2]),
        "mask_bit_identical": bool(np.array_equal(first[3], second[3])),
        "image_unmodified": bool(np.array_equal(image, image_backup)),
        "background_unmodified": bool(np.array_equal(background, background_backup)),
    }


def raw_arguments(runtime: GpuRuntime, image, background, mask) -> dict:
    """Build the full argument set for a direct call to the native export."""
    height, width = int(image.shape[0]), int(image.shape[1])
    return {
        "context": runtime._context,
        "image": image.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        "image_stride": int(image.strides[0]),
        "background": background.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        "background_stride": int(background.strides[0]),
        "width": width,
        "height": height,
        "sigma_multiplier": ctypes.c_double(SIGMA_MULTIPLIER),
        "threshold_floor": ctypes.c_double(THRESHOLD_FLOOR),
        "absolute_floor": ctypes.c_double(ABSOLUTE_FLOOR),
        "mad_scale": ctypes.c_double(MAD_SCALE),
        "candidate_value": int(CANDIDATE_VALUE),
        "out_residual_median": ctypes.byref(ctypes.c_float(0.0)),
        "out_mad": ctypes.byref(ctypes.c_float(0.0)),
        "out_threshold": ctypes.byref(ctypes.c_double(0.0)),
        "out_mask": mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        "out_mask_capacity": ctypes.c_longlong(int(mask.size)),
    }


def refusal_cases(runtime: GpuRuntime) -> dict:
    """Every malformed request must be refused with a non-zero code and no crash."""
    height, width = 32, 48
    image = np.full((height, width), 100.0, dtype=np.float32)
    image[10:14, 20:26] = 250.0
    background = cv2.GaussianBlur(image, (9, 9), 0.0)
    mask = np.full((height, width), 0xAB, dtype=np.uint8)
    function = runtime._dll.vf_cnr_mask_f32
    base = raw_arguments(runtime, image, background, mask)

    def call(**overrides) -> int:
        # Positional call: ctypes function pointers reject keyword arguments. The dict keeps every
        # ctypes object alive for the duration of the call.
        values = dict(base)
        values.update(overrides)
        return int(function(*(values[name] for name in ARG_ORDER)))

    cases = (
        ("null context", {"context": None}, CUDA_ERROR_INVALID_ARGUMENT),
        ("null image", {"image": None}, CUDA_ERROR_INVALID_ARGUMENT),
        ("null background", {"background": None}, CUDA_ERROR_INVALID_ARGUMENT),
        ("null out_residual_median", {"out_residual_median": None}, CUDA_ERROR_INVALID_ARGUMENT),
        ("null out_mad", {"out_mad": None}, CUDA_ERROR_INVALID_ARGUMENT),
        ("null out_threshold", {"out_threshold": None}, CUDA_ERROR_INVALID_ARGUMENT),
        ("null out_mask", {"out_mask": None}, CUDA_ERROR_INVALID_ARGUMENT),
        ("width 0", {"width": 0}, CUDA_ERROR_INVALID_ARGUMENT),
        ("height 0", {"height": 0}, CUDA_ERROR_INVALID_ARGUMENT),
        ("width negative", {"width": -5}, CUDA_ERROR_INVALID_ARGUMENT),
        ("height negative", {"height": -1}, CUDA_ERROR_INVALID_ARGUMENT),
        ("image stride too small", {"image_stride": width * 4 - 4}, CUDA_ERROR_INVALID_ARGUMENT),
        ("background stride too small",
         {"background_stride": width * 4 - 4}, CUDA_ERROR_INVALID_ARGUMENT),
        ("candidate_value 256", {"candidate_value": 256}, CUDA_ERROR_INVALID_ARGUMENT),
        ("candidate_value -1", {"candidate_value": -1}, CUDA_ERROR_INVALID_ARGUMENT),
        ("mask capacity one short",
         {"out_mask_capacity": ctypes.c_longlong(height * width - 1)}, CUDA_ERROR_INVALID_ARGUMENT),
        ("mask capacity zero",
         {"out_mask_capacity": ctypes.c_longlong(0)}, CUDA_ERROR_INVALID_ARGUMENT),
    )
    results = []
    mask_before = mask.copy()
    for name, overrides, expected in cases:
        code = call(**overrides)
        results.append({
            "case": name,
            "error_code": code,
            "refused": code != 0,
            "expected_error_code": expected,
            "as_expected": code == expected,
        })
    # The refusals must not have disturbed the context: a valid call still has to produce the
    # reference result, and the mask buffer must be untouched by the refused calls.
    device = device_call(runtime, image, background)
    reference = reference_cnr_steps(image, background)
    return {
        "cases": results,
        "all_refused": all(row["refused"] for row in results),
        "all_as_expected": all(row["as_expected"] for row in results),
        "valid_call_after_refusals_matches": bool(np.array_equal(device[3], reference[3])),
        "mask_buffer_untouched_by_refusals": bool(np.array_equal(mask, mask_before)),
    }


def _counter(runtime: GpuRuntime, function: str) -> dict:
    return runtime.performance_stats()["functions"].get(function, {})


def measure_timing(runtime: GpuRuntime) -> dict:
    """Device-event timing and PCIe bytes for the 2000x12000 ROI: new export vs the current path."""
    height, width = TIMING_SHAPE
    rng = np.random.default_rng(202_120)
    image = rng.uniform(0.0, 255.0, (height, width)).astype(np.float32)
    image[500:700, 3000:3400] = 255.0
    image[900:1000, 6000:6100] = 0.0
    background = cv2.GaussianBlur(image, (TIMING_KERNEL, TIMING_KERNEL), TIMING_SIGMA)
    residual = image - background
    plane_bytes = int(image.nbytes)

    # Warm up so the measurements exclude the first-touch cudaMalloc of the context buffers.
    device_call(runtime, image, background)
    runtime.median_f32(residual)

    # Current wiring as the detector drives it: the float32 Gaussian background on the device (one
    # upload of the image plus one download of the background), then two vf_median_f32 calls whose
    # operands are built on the host, then the threshold and the mask in NumPy.
    gaussian_before = dict(_counter(runtime, "vf_gaussian_blur_f32"))
    median_before = dict(_counter(runtime, "vf_median_f32"))
    baseline_times = []
    baseline_phases = []
    device_background = background
    for _ in range(TIMING_REPEATS):
        started = time.perf_counter()
        device_background = runtime.gaussian_blur_f32(image, TIMING_KERNEL, TIMING_SIGMA)
        gaussian_done = time.perf_counter()
        device_residual = image - device_background
        residual_done = time.perf_counter()
        residual_median = runtime.median_f32(device_residual)
        median_done = time.perf_counter()
        absdev = np.abs(device_residual - residual_median)
        absdev_done = time.perf_counter()
        mad = runtime.median_f32(absdev)
        mad_done = time.perf_counter()
        robust = max(MAD_SCALE * float(mad), ABSOLUTE_FLOOR)
        threshold = max(THRESHOLD_FLOOR, SIGMA_MULTIPLIER * robust)
        baseline_mask = (
            (np.abs(device_residual - residual_median) > threshold).astype(np.uint8)
            * CANDIDATE_VALUE
        )
        mask_done = time.perf_counter()
        baseline_times.append(mask_done - started)
        baseline_phases.append({
            "gaussian_device_ms": (gaussian_done - started) * 1e3,
            "host_residual_ms": (residual_done - gaussian_done) * 1e3,
            "median_residual_ms": (median_done - residual_done) * 1e3,
            "host_absdev_ms": (absdev_done - median_done) * 1e3,
            "median_mad_ms": (mad_done - absdev_done) * 1e3,
            "host_threshold_mask_ms": (mask_done - mad_done) * 1e3,
        })
    gaussian_after = dict(_counter(runtime, "vf_gaussian_blur_f32"))
    median_after = dict(_counter(runtime, "vf_median_f32"))
    baseline_reference = reference_cnr_steps(image, device_background)

    # New export: the operands are uploaded once each and every derived array stays on the device.
    cnr_before = dict(_counter(runtime, "vf_cnr_mask_f32"))
    new_times = []
    new_native = None
    device_result = None
    for _ in range(TIMING_REPEATS):
        started = time.perf_counter()
        device_result = device_call(runtime, image, background)
        new_times.append(time.perf_counter() - started)
        new_native = runtime.performance_stats()["native_timings_ms"]
    cnr_after = dict(_counter(runtime, "vf_cnr_mask_f32"))

    reference = reference_cnr_steps(image, background)
    per_repeat_gaussian = {
        key: (gaussian_after.get(key, 0) - gaussian_before.get(key, 0)) / TIMING_REPEATS
        for key in ("host_to_device_bytes", "device_to_host_bytes")
    }
    per_repeat_median = {
        key: (median_after.get(key, 0) - median_before.get(key, 0)) / TIMING_REPEATS
        for key in ("host_to_device_bytes", "device_to_host_bytes")
    }
    per_repeat_cnr = {
        key: (cnr_after.get(key, 0) - cnr_before.get(key, 0)) / TIMING_REPEATS
        for key in ("host_to_device_bytes", "device_to_host_bytes")
    }

    # The isolated median stage the task names: two vf_median_f32 calls plus the host mask, with no
    # Gaussian in the window.
    isolated_repeats = 3
    isolated_before = dict(_counter(runtime, "vf_median_f32"))
    isolated_times = []
    isolated_mask = None
    for _ in range(isolated_repeats):
        started = time.perf_counter()
        median_value = runtime.median_f32(residual)
        absdev = np.abs(residual - median_value)
        mad_value = runtime.median_f32(absdev)
        robust = max(MAD_SCALE * float(mad_value), ABSOLUTE_FLOOR)
        threshold = max(THRESHOLD_FLOOR, SIGMA_MULTIPLIER * robust)
        isolated_mask = (
            (np.abs(residual - median_value) > threshold).astype(np.uint8) * CANDIDATE_VALUE
        )
        isolated_times.append(time.perf_counter() - started)
    isolated_after = dict(_counter(runtime, "vf_median_f32"))
    isolated_h2d = (isolated_after.get("host_to_device_bytes", 0)
                    - isolated_before.get("host_to_device_bytes", 0)) / isolated_repeats
    isolated_d2h = (isolated_after.get("device_to_host_bytes", 0)
                    - isolated_before.get("device_to_host_bytes", 0)) / isolated_repeats

    return {
        "shape": list(TIMING_SHAPE),
        "kernel": TIMING_KERNEL,
        "sigma": TIMING_SIGMA,
        "repeats": TIMING_REPEATS,
        "plane_bytes": plane_bytes,
        "new_export": {
            "wall_ms": [round(value * 1e3, 3) for value in new_times],
            "wall_min_ms": round(min(new_times) * 1e3, 3),
            "wall_median_ms": round(float(np.median(new_times)) * 1e3, 3),
            "native_timings_ms": new_native,
            "host_to_device_bytes": per_repeat_cnr["host_to_device_bytes"],
            "device_to_host_bytes": per_repeat_cnr["device_to_host_bytes"],
            "mask_matches_reference": bool(np.array_equal(device_result[3], reference[3])),
            "median_exact": scalar_state(device_result[0], reference[0]) != "mismatch",
            "mad_exact": scalar_state(device_result[1], reference[1]) != "mismatch",
            "threshold_exact": abs(float(device_result[2]) - float(reference[2])) == 0.0,
        },
        "current_stage_with_device_gaussian": {
            "wall_ms": [round(value * 1e3, 3) for value in baseline_times],
            "wall_min_ms": round(min(baseline_times) * 1e3, 3),
            "wall_median_ms": round(float(np.median(baseline_times)) * 1e3, 3),
            "phases_ms": [
                {key: round(value, 3) for key, value in row.items()} for row in baseline_phases
            ],
            "phase_median_ms": {
                key: round(float(np.median([row[key] for row in baseline_phases])), 3)
                for key in baseline_phases[0]
            },
            "gaussian_host_to_device_bytes": per_repeat_gaussian["host_to_device_bytes"],
            "gaussian_device_to_host_bytes": per_repeat_gaussian["device_to_host_bytes"],
            "median_host_to_device_bytes": per_repeat_median["host_to_device_bytes"],
            "median_device_to_host_bytes": per_repeat_median["device_to_host_bytes"],
            # The mask must reproduce the reference for the background the device itself produced;
            # the device Gaussian is only equal to cv2 within the documented tolerance.
            "mask_matches_reference_of_same_background": bool(
                np.array_equal(baseline_mask, baseline_reference[3])
            ),
        },
        "isolated_median_stage": {
            "wall_ms": [round(value * 1e3, 3) for value in isolated_times],
            "wall_min_ms": round(min(isolated_times) * 1e3, 3),
            "wall_median_ms": round(float(np.median(isolated_times)) * 1e3, 3),
            "host_to_device_bytes": isolated_h2d,
            "device_to_host_bytes": isolated_d2h,
            "mask_matches_reference": bool(np.array_equal(isolated_mask, reference[3])),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", default=str(ROOT / "gpu" / "visionflow_cuda.dll"))
    parser.add_argument("--skip-timing", action="store_true",
                        help="Skip the 2000x12000 timing block (the equivalence sweep still runs).")
    args = parser.parse_args()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    runtime = GpuRuntime(args.dll)
    if not runtime.available:
        print(f"CUDA runtime unavailable: {runtime.unavailable_reason}")
        return 2
    if not runtime.supports_cnr_mask_f32:
        print("CUDA DLL has no vf_cnr_mask_f32 export; rebuild with gpu\\build_cuda_dll.ps1")
        return 2

    lines: list[str] = []
    record: dict = {
        "device": runtime.device_name,
        "compute_capability": runtime.compute_capability,
        "capabilities": runtime.status(True)["capabilities"],
        "open_cv": cv2.__version__,
        "numpy": np.__version__,
        "parameters": {
            "sigma_multiplier": SIGMA_MULTIPLIER,
            "threshold_floor": THRESHOLD_FLOOR,
            "absolute_floor": ABSOLUTE_FLOOR,
            "mad_scale": MAD_SCALE,
            "candidate_value": CANDIDATE_VALUE,
        },
        "claimed_tolerance": {
            "residual_median": "bit-exact (no tolerance)",
            "mad": "bit-exact (no tolerance)",
            "threshold": "exact double equality (no tolerance)",
            "mask": "byte-exact (no tolerance)",
        },
        "sweep_plan": {
            "sizes": [list(shape) for shape in SIZES],
            "kinds": list(KINDS),
            "full_pairs": [list(pair) for pair in SWEEP],
            "production_pairs": [list(pair) for pair in PRODUCTION_SWEEP],
            "production_sizes": [list(shape) for shape in PRODUCTION_SIZES],
        },
    }

    lines.append("== vf_cnr_mask_f32 equivalence evidence ==")
    lines.append(f"device: {runtime.device_name} (sm {runtime.compute_capability})")
    lines.append(f"openCV: cv2 {cv2.__version__}, numpy {np.__version__}")
    lines.append("reference: detectors/detector_202_1.py::_automatic_cnr_mask steps after the cv2 Gaussian")
    lines.append(
        "parameters: sigma_multiplier=3.0, threshold_floor=8.0, absolute_floor=1e-6, "
        "mad_scale=1.4826, candidate_value=255 (detector defaults)"
    )

    print("sweeping scenes ...", flush=True)
    sweep = sweep_cases(runtime)
    record["sweep"] = sweep
    lines.append("")
    lines.append(
        f"[scene x (kernel, sigma) sweep] {sweep['cases']} cases "
        f"({len(SIZES)} shapes x {len(KINDS)} contents; full {len(SWEEP)} pairs for the cheap "
        f"shapes, {len(PRODUCTION_SWEEP)} pairs for {list(PRODUCTION_SIZES)})"
    )
    lines.append(
        f"  residual_median: bit-exact {sweep['median_exact']}, both-NaN {sweep['median_nan']}, "
        f"mismatch {sweep['median_mismatch']}"
    )
    lines.append(
        f"  mad:             bit-exact {sweep['mad_exact']}, both-NaN {sweep['mad_nan']}, "
        f"mismatch {sweep['mad_mismatch']}"
    )
    lines.append(
        f"  threshold:       exact {sweep['threshold_exact']}, mismatch {sweep['threshold_mismatch']}"
        f", worst |diff| = {sweep['threshold_max_abs_diff']:.3e}"
    )
    lines.append(
        f"  mask:            byte-identical {sweep['mask_identical']}, "
        f"mismatch {sweep['mask_mismatch']} "
        f"({sweep['mask_differing_pixels']} differing pixels in total)"
    )
    for row in sweep["per_size"]:
        lines.append(
            f"    shape={str(tuple(row['shape'])):<15} pairs={row['pairs']:>2} "
            f"cases={row['cases']:>3} failures={row['failures']}"
        )
    if sweep["failures"]:
        lines.append("  first failing cases:")
        for failure in sweep["failures"]:
            lines.append(f"    {json.dumps(failure, sort_keys=True)}")

    print("threshold rounding probe ...", flush=True)
    probe = threshold_rounding_probe(runtime)
    record["threshold_rounding_probe"] = probe
    lines.append("")
    lines.append("[threshold rounding probe] background=0, residual multiset pinned to median=0, mad=2.0")
    lines.append(
        f"  threshold double {probe['threshold_double']!r} -> float32 "
        f"{probe['threshold_float32']!r} (rounds up: {probe['threshold_rounds_up_to_float32']})"
    )
    lines.append(
        f"  device mask ones {probe['device_mask_ones']}, python mask ones "
        f"{probe['python_mask_ones']}, double-comparison mask ones "
        f"{probe['double_comparison_mask_ones']}"
    )
    lines.append(
        f"  probe discriminates float32 from double comparison: {probe['probe_is_discriminating']}; "
        f"mask identical: {probe['mask_identical']}"
    )

    print("result contract ...", flush=True)
    contract = result_contract(runtime)
    record["result_contract"] = contract
    lines.append("")
    lines.append("[bridge result contract] the mapping the detector's call site reads")
    lines.append(
        f"  type={contract['type']} keys={contract['keys']} "
        f"has_required_keys={contract['has_required_keys']} "
        f"exact_key_set={contract['exact_key_set']}"
    )
    lines.append(
        f"  median={contract['median_type']} mad={contract['mad_type']} "
        f"threshold={contract['threshold_type']} mask={contract['mask_type']} "
        f"mask shape={contract['mask_shape']}"
    )

    print("stride ROI case ...", flush=True)
    stride = stride_case(runtime)
    record["non_contiguous_roi"] = stride
    lines.append("")
    lines.append("[non-contiguous ROI] a strided view of a wider plane, passed by byte stride")
    lines.append(
        f"  strides={tuple(stride['view_strides'])} non-contiguous={stride['non_contiguous']} "
        f"median/mad/threshold exact="
        f"{stride['median_exact'] and stride['mad_exact'] and stride['threshold_exact']} "
        f"mask identical={stride['mask_identical']}"
    )

    print("determinism and non-mutation ...", flush=True)
    determinism = determinism_and_non_mutation(runtime)
    record["determinism"] = determinism
    lines.append("")
    lines.append("[determinism / non-mutation]")
    lines.append(
        f"  two identical calls: median {determinism['median_bit_identical']}, "
        f"mad {determinism['mad_bit_identical']}, threshold {determinism['threshold_identical']}, "
        f"mask {determinism['mask_bit_identical']}"
    )
    lines.append(
        f"  inputs unmodified: image {determinism['image_unmodified']}, "
        f"background {determinism['background_unmodified']}"
    )

    print("refusal cases ...", flush=True)
    refusals = refusal_cases(runtime)
    record["refusals"] = refusals
    lines.append("")
    lines.append("[refusals] malformed requests must return a non-zero code without crashing")
    for row in refusals["cases"]:
        lines.append(
            f"    {row['case']:<32} error_code={row['error_code']:<3} "
            f"refused={row['refused']} as_expected={row['as_expected']}"
        )
    lines.append(f"  all refused: {refusals['all_refused']}")
    lines.append(f"  all with the documented code: {refusals['all_as_expected']}")
    lines.append(
        f"  a valid call after the refusals still matches the reference: "
        f"{refusals['valid_call_after_refusals_matches']}"
    )
    lines.append(
        f"  the refused calls left the mask buffer untouched: "
        f"{refusals['mask_buffer_untouched_by_refusals']}"
    )

    timing = None
    if not args.skip_timing:
        print(f"timing {TIMING_SHAPE} ...", flush=True)
        timing = measure_timing(runtime)
        record["timing"] = timing
        new = timing["new_export"]
        current = timing["current_stage_with_device_gaussian"]
        isolated = timing["isolated_median_stage"]
        plane = timing["plane_bytes"]
        current_h2d = (current["gaussian_host_to_device_bytes"]
                       + current["median_host_to_device_bytes"])
        current_d2h = (current["gaussian_device_to_host_bytes"]
                       + current["median_device_to_host_bytes"])
        lines.append("")
        lines.append(
            f"[timing] {timing['shape'][0]}x{timing['shape'][1]} ROI, kernel {timing['kernel']}, "
            f"sigma {timing['sigma']}, {timing['repeats']} repeats, plane = {plane} bytes "
            f"({plane / 2**20:.1f} MiB)"
        )
        lines.append(
            f"  new export vf_cnr_mask_f32: wall min {new['wall_min_ms']:.1f} ms / median "
            f"{new['wall_median_ms']:.1f} ms; device events {json.dumps(new['native_timings_ms'])}"
        )
        lines.append(
            f"    host-to-device {new['host_to_device_bytes'] / 2**20:.1f} MiB, "
            f"device-to-host {new['device_to_host_bytes'] / 2**20:.1f} MiB "
            f"(2 operand planes up, 1 mask plane down; the medians and the threshold are written by "
            f"the host)"
        )
        lines.append(
            f"  current stage (device Gaussian + 2 x vf_median_f32 + host mask): wall min "
            f"{current['wall_min_ms']:.1f} ms / median {current['wall_median_ms']:.1f} ms"
        )
        lines.append(
            f"    host-to-device {current_h2d / 2**20:.1f} MiB, device-to-host "
            f"{current_d2h / 2**20:.1f} MiB (Gaussian image up "
            f"{current['gaussian_host_to_device_bytes'] / 2**20:.1f} MiB + background down "
            f"{current['gaussian_device_to_host_bytes'] / 2**20:.1f} MiB, then the residual and its "
            f"absolute deviation up {current['median_host_to_device_bytes'] / 2**20:.1f} MiB)"
        )
        lines.append(
            "    phase medians (ms): "
            + ", ".join(f"{key}={value:.1f}" for key, value in current["phase_median_ms"].items())
        )
        lines.append(
            f"  isolated median stage (2 x vf_median_f32 + host mask, the task's baseline): wall min "
            f"{isolated['wall_min_ms']:.1f} ms / median {isolated['wall_median_ms']:.1f} ms; "
            f"host-to-device {isolated['host_to_device_bytes'] / 2**20:.1f} MiB, device-to-host "
            f"{isolated['device_to_host_bytes']:.1f} B"
        )
        lines.append(
            f"  host-to-device comparison: new export {new['host_to_device_bytes'] / 2**20:.1f} MiB "
            f"vs isolated baseline {isolated['host_to_device_bytes'] / 2**20:.1f} MiB "
            f"vs current stage {current_h2d / 2**20:.1f} MiB"
        )

    claim_ok = bool(
        sweep["median_mismatch"] == 0
        and sweep["mad_mismatch"] == 0
        and sweep["threshold_mismatch"] == 0
        and sweep["mask_mismatch"] == 0
        and sweep["cases"] > 0
        and probe["mask_identical"]
        and probe["median_exact"]
        and probe["mad_exact"]
        and probe["threshold_exact"]
        and probe["premise_median_is_zero"]
        and probe["premise_mad_is_two"]
        and probe["probe_is_discriminating"]
        and stride["median_exact"] and stride["mad_exact"] and stride["threshold_exact"]
        and stride["mask_identical"]
        and contract["has_required_keys"]
        and contract["exact_key_set"]
        and all(bool(value) for value in determinism.values())
        and refusals["all_refused"]
        and refusals["all_as_expected"]
        and refusals["valid_call_after_refusals_matches"]
        and refusals["mask_buffer_untouched_by_refusals"]
    )
    if timing is not None:
        claim_ok = claim_ok and bool(
            timing["new_export"]["mask_matches_reference"]
            and timing["new_export"]["median_exact"]
            and timing["new_export"]["mad_exact"]
            and timing["new_export"]["threshold_exact"]
            and timing["current_stage_with_device_gaussian"][
                "mask_matches_reference_of_same_background"]
            and timing["isolated_median_stage"]["mask_matches_reference"]
        )
    record["claim_ok"] = claim_ok
    lines.append("")
    lines.append("[claimed tolerance] no tolerance is claimed; every compared field must be exact")
    lines.append(
        "  residual_median bit-exact, mad bit-exact, threshold double-exact, mask byte-exact, "
        f"every refusal honoured: {claim_ok}"
    )
    if not claim_ok:
        lines.append("  FAILED: see the failing cases above")

    EVIDENCE_TEXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    EVIDENCE_JSON.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {EVIDENCE_TEXT.relative_to(ROOT)}")
    print(f"written: {EVIDENCE_JSON.relative_to(ROOT)}")
    runtime.close()
    return 0 if claim_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
