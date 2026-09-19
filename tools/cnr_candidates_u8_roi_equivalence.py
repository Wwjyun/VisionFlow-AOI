"""Validate the resident 202 candidate export (mask, components and ring CNR on the device).

For every scene and parameter variant the detector runs three ways on the same image:

* ``device``  - ``vf_cnr_candidates_u8_roi`` keeps morphology, exclusion, connected components and
  the ring CNR statistics on the GPU and downloads one record per candidate;
* ``hybrid``  - the previous production route: the resident CNR mask export, then OpenCV
  connected components and NumPy ring statistics on the host;
* ``cpu``     - the CPU reference detector.

``device`` must equal ``hybrid`` in every defect field (both read the same device mask), and must
equal ``cpu`` in every field except the four residual diagnostics that the float32 Gaussian is
documented to drift and the backend provenance labels. Unsupported parameters and rings that need
the whole-image background must fall back to the host path with unchanged results. Evidence is
written under ``outputs_validation/cnr_candidates``.

Usage:
    .\\env\\Scripts\\python.exe tools\\cnr_candidates_u8_roi_equivalence.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import PropertyMock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gpu_runtime import GpuRuntime  # noqa: E402
from detectors.detector_202_1 import Detector202_1  # noqa: E402

OUT_DIR = ROOT / "outputs_validation" / "cnr_candidates"
DRIFTING = {"mad", "residual_median", "residual_threshold", "robust_noise_sigma"}
PROVENANCE = {"background_backend", "residual_backend", "background_precision_note", "component_backend"}

# The production center/edge masks cover most of a small ROI, so small scenes start from masks off
# and the mask variants switch them back on; otherwise most variants would compare empty lists.
SMALL_BASE = {"center_mask_enabled": False, "edge_mask_enabled": False}
VARIANTS = {
    "masks_off": {},
    "production_masks": {"center_mask_enabled": True, "edge_mask_enabled": True},
    "connectivity_4": {"connectivity": 4},
    "close_k5_i2": {"morph_operation": "close", "morph_kernel": 5, "morph_iterations": 2},
    "dilate_k3": {"morph_operation": "dilate"},
    "erode_k3": {"morph_operation": "erode", "min_component_area_px": 1},
    "no_morphology": {"morph_iterations": 0},
    "custom_center_all_insets": {
        "center_mask_enabled": True, "edge_mask_enabled": True,
        "center_mask_use_image_center": False, "center_mask_x": 90, "center_mask_y": 70,
        "center_mask_width": 25, "center_mask_height": 40, "edge_inset_all": 9,
    },
    "center_only_offset": {
        "center_mask_enabled": True, "center_mask_use_image_center": False,
        "center_mask_x": 40, "center_mask_y": 110, "center_mask_width": 60, "center_mask_height": 30,
    },
    "tight_padding": {
        "background_padding_min_px": 2, "background_padding_max_px": 5, "background_padding_scale": 0.5,
    },
    "wide_padding_margin_0": {
        "background_padding_scale": 3.0, "background_padding_max_px": 80, "component_border_margin_px": 0,
    },
    "small_components": {"min_component_area_px": 1, "max_component_area_px": 40},
    "candidate_value_1": {"candidate_max_value": 1},
    "min_background_0": {"min_background_pixels": 0},
}
# Parameters the device export must decline, leaving the host path in charge.
FALLBACK_VARIANTS = {
    "even_morph_kernel": {"morph_kernel": 4},
    "global_background_needed": {"min_background_pixels": 10 ** 9},
}


def make_scene(height: int, width: int, channels: int, seed: int, defects: int) -> np.ndarray:
    """Noisy textured surface with blobs, 8-only diagonal chains, nested rings and touching pairs."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    base = 120.0 + 18.0 * np.sin(x / 57.0) + 12.0 * np.cos(y / 91.0) + rng.normal(0.0, 4.0, (height, width))
    gray = base.clip(0, 255).astype(np.uint8)
    for index in range(defects):
        cx = int(rng.integers(6, width - 6))
        cy = int(rng.integers(6, height - 6))
        value = int(rng.choice([15, 35, 230, 250]))
        kind = index % 5
        if kind == 0:
            axes = (int(rng.integers(1, 9)), int(rng.integers(1, 9)))
            cv2.ellipse(gray, (cx, cy), axes, float(rng.integers(0, 180)), 0, 360, value, -1)
        elif kind == 1:
            length = int(rng.integers(4, 30))
            for step in range(length):
                px, py = cx + step, cy + step
                if px < width and py < height:
                    gray[py, px] = value
        elif kind == 2:
            radius = int(rng.integers(5, 14))
            cv2.circle(gray, (cx, cy), radius, value, 2)
            cv2.circle(gray, (cx, cy), max(1, radius // 4), value, -1)
        elif kind == 3:
            cv2.rectangle(gray, (cx - 3, cy - 2), (cx + 3, cy + 2), value, -1)
            cv2.rectangle(gray, (cx + 4, cy - 1), (cx + 7, cy + 1), 255 - value, -1)
        else:
            cv2.line(gray, (cx, cy), (min(width - 1, cx + 25), max(0, cy - 9)), value, 1)
    if channels == 1:
        return gray
    return np.dstack((gray, np.roll(gray, 1, axis=1), np.roll(gray, 1, axis=0)))


def make_regular_array(height: int, width: int, channels: int) -> np.ndarray:
    """Identical dark squares on a flat surface: CNR ties that only the bbox tie-break can order."""
    gray = np.full((height, width), 128, dtype=np.uint8)
    gray[::7, ::5] = 131
    for row in range(20, height - 20, 24):
        for col in range(20, width - 20, 24):
            gray[row : row + 4, col : col + 4] = 40
    if channels == 1:
        return gray
    return np.dstack((gray, gray, gray))


def defects_without(defects: list[dict], ignored: set[str]) -> list[dict]:
    cleaned = []
    for defect in defects:
        row = dict(defect)
        row["metadata"] = {key: value for key, value in defect.get("metadata", {}).items() if key not in ignored}
        cleaned.append(row)
    return cleaned


def run_detector(runtime, image, box, params, *, use_gpu, candidates_export=True):
    x, y, width, height = box
    tile = image[y : y + height, x : x + width]
    detector = Detector202_1(params=params, use_gpu=use_gpu, gpu_runtime=runtime if use_gpu else None)
    device_roi = runtime.upload_image(image).roi(x, y, width, height) if use_gpu else None
    if use_gpu and not candidates_export:
        with patch.object(
            GpuRuntime, "supports_cnr_candidates_u8_roi", new_callable=PropertyMock, return_value=False
        ):
            started = time.perf_counter()
            result = detector.run(tile, device_roi=device_roi)
    else:
        started = time.perf_counter()
        result = detector.run(tile, device_roi=device_roi)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return result, elapsed_ms


def component_backend(result: dict) -> str:
    backends = {defect["metadata"]["component_backend"] for defect in result["defects"]}
    return ",".join(sorted(backends)) if backends else "(no defects)"


def compare_case(runtime, image, box, name, params, expect_device: bool) -> dict:
    device, device_ms = run_detector(runtime, image, box, params, use_gpu=True)
    hybrid, hybrid_ms = run_detector(runtime, image, box, params, use_gpu=True, candidates_export=False)
    cpu, cpu_ms = run_detector(runtime, image, box, params, use_gpu=False)
    device_defects = defects_without(device["defects"], {"component_backend"})
    hybrid_defects = defects_without(hybrid["defects"], {"component_backend"})
    cpu_defects = defects_without(cpu["defects"], DRIFTING | PROVENANCE)
    device_vs_cpu = defects_without(device["defects"], DRIFTING | PROVENANCE)
    backend = component_backend(device)
    # A comparison of two empty defect lists proves nothing, so an empty case counts as a failure.
    routed_as_expected = bool(device["defects"]) and (
        backend == "cuda_resident" if expect_device else backend == "opencv_cpu"
    )
    return {
        "name": name,
        "image_shape": list(image.shape),
        "roi": list(box),
        "defects": len(device["defects"]),
        "pass": device["pass"],
        "component_backend": backend,
        "routed_as_expected": routed_as_expected,
        "device_equals_hybrid": device_defects == hybrid_defects and device["pass"] == hybrid["pass"],
        "device_equals_cpu_except_drift": device_vs_cpu == cpu_defects and device["pass"] == cpu["pass"],
        "device_ms": device_ms,
        "hybrid_ms": hybrid_ms,
        "cpu_ms": cpu_ms,
    }


def component_count_case(runtime, image, box, params) -> dict:
    """The export's raw component count must equal OpenCV's on the hybrid path's final mask."""
    x, y, width, height = box
    tile = image[y : y + height, x : x + width]
    detector = Detector202_1(params=params, use_gpu=True, gpu_runtime=runtime)
    roi = runtime.upload_image(image).roi(x, y, width, height)
    detector._active_device_roi = roi
    with patch.object(GpuRuntime, "supports_cnr_candidates_u8_roi", new_callable=PropertyMock, return_value=False):
        analysis = detector._automatic_cnr_mask(detector._make_gray(tile))
    labels, _, _, _ = cv2.connectedComponentsWithStats(
        analysis["candidate_mask"], connectivity=int(detector.params.get("connectivity", 8))
    )
    int_params, real_params = detector._device_candidate_parameters(height, width)
    device = runtime.cnr_candidates_u8_roi(roi, int_params, real_params)
    return {"opencv_components": int(labels) - 1, "device_components": int(device["component_count"])}


def main() -> int:
    runtime = GpuRuntime(ROOT / "gpu" / "visionflow_cuda.dll", fallback_to_cpu=True)
    if not runtime.available or not runtime.supports_cnr_candidates_u8_roi:
        raise SystemExit("vf_cnr_candidates_u8_roi is unavailable; rebuild the CUDA DLL")
    cases = []
    counts = []
    try:
        for channels in (1, 3):
            for seed in (3, 11):
                image = make_scene(260, 320, channels, seed + channels, defects=70)
                box = (12, 9, 280, 220)
                for name, overrides in VARIANTS.items():
                    params = {**SMALL_BASE, **overrides}
                    cases.append(compare_case(runtime, image, box, f"{name}/c{channels}/s{seed}", params, True))
                for name, overrides in FALLBACK_VARIANTS.items():
                    params = {**SMALL_BASE, **overrides}
                    cases.append(compare_case(runtime, image, box, f"{name}/c{channels}/s{seed}", params, False))
                for overrides in ({}, {"connectivity": 4}, {"morph_iterations": 0}):
                    counts.append(component_count_case(runtime, image, box, {**SMALL_BASE, **overrides}))
        dense = make_scene(400, 300, 3, 99, defects=900)
        cases.append(compare_case(
            runtime, dense, (0, 0, 300, 400), "dense_defects", {**SMALL_BASE, "min_component_area_px": 1}, True))
        for channels in (1, 3):
            regular = make_regular_array(240, 320, channels)
            cases.append(compare_case(
                runtime, regular, (0, 0, 320, 240), f"regular_array_ties/c{channels}", dict(SMALL_BASE), True))
        production = make_scene(12000, 2000, 3, 20260915, defects=160)
        for repeat in range(3):
            cases.append(compare_case(runtime, production, (0, 0, 2000, 12000), f"production_12000x2000/r{repeat}", {}, True))
        counts.append(component_count_case(runtime, production, (0, 0, 2000, 12000), {}))
        metrics = runtime.performance_stats()
    finally:
        runtime.close()

    production_rows = [row for row in cases if row["name"].startswith("production_12000x2000")]
    passed = (
        all(row["device_equals_hybrid"] and row["device_equals_cpu_except_drift"] and row["routed_as_expected"]
            for row in cases)
        and all(row["opencv_components"] == row["device_components"] for row in counts)
    )
    payload = {
        "passed": passed,
        "case_count": len(cases),
        "device_equals_hybrid": sum(row["device_equals_hybrid"] for row in cases),
        "device_equals_cpu_except_drift": sum(row["device_equals_cpu_except_drift"] for row in cases),
        "routed_as_expected": sum(row["routed_as_expected"] for row in cases),
        "component_count_matches": sum(row["opencv_components"] == row["device_components"] for row in counts),
        "component_count_cases": len(counts),
        "production_defects": production_rows[0]["defects"] if production_rows else 0,
        "production_device_median_ms": statistics.median(row["device_ms"] for row in production_rows),
        "production_hybrid_median_ms": statistics.median(row["hybrid_ms"] for row in production_rows),
        "production_cpu_median_ms": statistics.median(row["cpu_ms"] for row in production_rows),
        "cases": cases,
        "component_counts": counts,
        "metrics": metrics,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "cnr_candidates_u8_roi_equivalence.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"passed={passed}",
        f"cases={payload['case_count']}",
        f"device_equals_hybrid={payload['device_equals_hybrid']}/{payload['case_count']}",
        f"device_equals_cpu_except_drift={payload['device_equals_cpu_except_drift']}/{payload['case_count']}",
        f"routed_as_expected={payload['routed_as_expected']}/{payload['case_count']}",
        f"component_count_matches={payload['component_count_matches']}/{payload['component_count_cases']}",
        f"production_defects={payload['production_defects']}",
        f"production_detector_median_ms device={payload['production_device_median_ms']:.1f} "
        f"hybrid={payload['production_hybrid_median_ms']:.1f} cpu={payload['production_cpu_median_ms']:.1f}",
    ]
    failures = [row for row in cases if not (
        row["device_equals_hybrid"] and row["device_equals_cpu_except_drift"] and row["routed_as_expected"])]
    lines += [f"FAIL {row['name']}: {row}" for row in failures[:10]]
    lines += [f"COUNT MISMATCH {row}" for row in counts if row["opencv_components"] != row["device_components"]]
    text = "\n".join(lines) + "\n"
    (OUT_DIR / "cnr_candidates_u8_roi_equivalence.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
