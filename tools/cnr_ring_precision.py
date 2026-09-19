"""Measure how far 202-CS-SN-1's float32 ring statistics sit from a float64 recomputation.

The GPU port of the connected-components / ring-CNR stage will sum the same
pixels in a different order.  Before that work starts we need to know how large
the resulting CNR deviation actually is and how close real candidates sit to the
NG threshold, because only the second number decides whether a summation-order
change can flip a PASS/NG decision.

Usage:
    .\\env\\Scripts\\python.exe tools\\cnr_ring_precision.py [--json OUTPUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detectors.detector_202_1 import Detector202_1  # noqa: E402
from tests.test_detector_202_1_cnr_contract import (  # noqa: E402
    _cnr_scene,
    _detector,
    _float64_ring_reference,
)


def _measure(seed: int, extra: dict, size: tuple[int, int]) -> dict:
    gray = _cnr_scene(seed, height=size[0], width=size[1])
    detector = _detector(**extra)
    analysis = detector._automatic_cnr_mask(gray)
    candidates = detector._collect_candidates(
        analysis["image_float"],
        analysis["candidate_mask"],
        analysis["inclusion_mask"],
    )
    connectivity = int(extra.get("connectivity", 8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        analysis["candidate_mask"], connectivity=connectivity
    )
    by_box = {}
    for label in range(1, stats.shape[0]):
        x, y, component_width, component_height, area = (
            int(value) for value in stats[label]
        )
        if area < analysis["min_area"]:
            continue
        by_box[(x, y, component_width, component_height)] = (
            _float64_ring_reference(
                analysis["image_float"],
                labels,
                stats,
                analysis["inclusion_mask"],
                label,
                gray.shape[0],
                gray.shape[1],
                detector.params,
            )
        )

    cnr_values = sorted(candidate.cnr for candidate in candidates)
    smallest_gap = float("inf")
    for first, second in zip(cnr_values, cnr_values[1:]):
        gap = second - first
        if gap > 0.0:
            smallest_gap = min(smallest_gap, gap)
    worst_cnr = 0.0
    worst_contrast = 0.0
    worst_relative = 0.0
    rows = []
    for candidate in candidates:
        reference = by_box[candidate.bbox]
        cnr_error = abs(candidate.cnr - reference["cnr"])
        contrast_error = abs(candidate.contrast - reference["contrast"])
        worst_cnr = max(worst_cnr, cnr_error)
        worst_contrast = max(worst_contrast, contrast_error)
        if reference["cnr"] > 0:
            worst_relative = max(worst_relative, cnr_error / reference["cnr"])
        rows.append(
            {
                "bbox": list(candidate.bbox),
                "cnr_float32": candidate.cnr,
                "cnr_float64": reference["cnr"],
                "cnr_abs_error": cnr_error,
                "contrast_abs_error": contrast_error,
                "background_area_px": candidate.background_area,
            }
        )
    return {
        "seed": seed,
        "shape": [int(size[0]), int(size[1])],
        "extra": {key: str(value) for key, value in sorted(extra.items())},
        "candidate_count": len(candidates),
        "worst_cnr_abs_error": worst_cnr,
        "worst_cnr_relative_error": worst_relative,
        "worst_contrast_abs_error": worst_contrast,
        "smallest_distinct_cnr_gap": smallest_gap,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--seeds", type=int, default=8)
    args = parser.parse_args()

    cases = [(seed, {}, (320, 480)) for seed in range(201, 201 + args.seeds)]
    cases += [
        (300, {"min_background_pixels": 20}, (256, 256)),
        (301, {"background_kernel_size": 21, "gaussian_sigma": 1.5}, (320, 480)),
        (302, {"cnr_noise_floor": 0.5}, (320, 480)),
        (303, {"morph_operation": "none", "morph_iterations": 0}, (320, 480)),
        (304, {"connectivity": 4}, (320, 480)),
        (305, {"min_component_area_px": 5, "max_component_area_ratio": 0.05}, (512, 512)),
        (306, {}, (768, 1024)),
    ]

    reports = [_measure(seed, extra, size) for seed, extra, size in cases]
    total = sum(report["candidate_count"] for report in reports)
    worst_cnr = max(report["worst_cnr_abs_error"] for report in reports)
    worst_relative = max(report["worst_cnr_relative_error"] for report in reports)
    worst_contrast = max(report["worst_contrast_abs_error"] for report in reports)
    smallest_gap = min(
        report["smallest_distinct_cnr_gap"] for report in reports
    )

    print("202-CS-SN-1 ring CNR: float32 (production) vs float64 recomputation")
    print(f"scenes: {len(reports)}   candidates: {total}")
    print(f"worst |cnr_deviation|       : {worst_cnr:.6e}")
    print(f"worst  cnr relative error   : {worst_relative:.6e}")
    print(f"worst |contrast_deviation|  : {worst_contrast:.6e}")
    print(f"smallest distinct CNR gap   : {smallest_gap:.6e}")
    print(
        "  (a summation-order change larger than this gap can reorder "
        "candidates)"
    )
    print(
        "  deviation-to-gap ratio     : "
        + (
            "n/a"
            if smallest_gap in (0.0, float("inf"))
            else f"{worst_cnr / smallest_gap:.3e}"
        )
    )
    print(
        "  NOTE: Detector202_1.detect applies no CNR threshold - every candidate "
        "becomes a defect - so CNR only affects metadata and candidate order."
    )

    payload = {
        "kind": "cnr_ring_precision",
        "summary": {
            "scene_count": len(reports),
            "candidate_count": total,
            "worst_cnr_abs_error": worst_cnr,
            "worst_cnr_relative_error": worst_relative,
            "worst_contrast_abs_error": worst_contrast,
            "smallest_distinct_cnr_gap": smallest_gap,
            "detector_applies_cnr_threshold": False,
        },
        "scenes": reports,
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
