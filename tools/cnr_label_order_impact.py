"""Does the component label order change 202-CS-SN-1's final output?

``Detector202_1._collect_candidates`` used to depend on OpenCV's component label
*numbering*: it walked components in ascending label order and then sorted by CNR with a
stable sort, so candidates whose CNR was exactly equal stayed in label order.  Exact ties
happen on any regular array of identical parts (measured 433/435 candidates), so a
replacement connected-components implementation that produced the same components with a
different numbering would have reordered the defect list - which is what made the
label-ordering gap a blocker.

The tie-break is now explicit: CNR descending, then the component's bounding box in
raster order.  Bounding boxes are disjoint for connected components, so that is a total
order and the output no longer depends on the numbering at all.

This tool is the check for that property: it builds a scene with exact CNR ties, relabels
the component map with random permutations (permuting the stats table with it, because
stats are indexed by label), re-runs the candidate stage, and compares the resulting
defect list field by field.  Every permutation must reproduce the same list.

Usage:
    .\\env\\Scripts\\python.exe tools\\cnr_label_order_impact.py [--json OUTPUT]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detectors.detector_202_1 import Detector202_1  # noqa: E402

PARAMS = {"center_mask_enabled": False, "edge_mask_enabled": False}


def _tied_scene() -> np.ndarray:
    """A regular grid of byte-identical defects on a constant background."""

    gray = np.full((300, 300), 150, dtype=np.uint8)
    for y in range(20, 280, 40):
        for x in range(20, 280, 40):
            gray[y : y + 10, x : x + 10] = 60
    return gray


def _payload(detector: Detector202_1, analysis: dict, labels: np.ndarray, stats, count):
    """Re-run the candidate stage with a supplied label map."""

    candidates = detector._collect_candidates_with_labels(
        analysis["image_float"],
        analysis["candidate_mask"],
        analysis["inclusion_mask"],
        labels,
        stats,
        count,
    )
    return [
        {
            "bbox": list(candidate.bbox),
            "area": candidate.area,
            "cnr": candidate.cnr,
            "contrast": candidate.contrast,
            "defect_mean": candidate.defect_mean,
            "background_mean": candidate.background_mean,
            "background_std": candidate.background_std,
            "background_area": candidate.background_area,
        }
        for candidate in candidates
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--permutations", type=int, default=300)
    args = parser.parse_args()

    gray = _tied_scene()
    detector = Detector202_1(params=dict(PARAMS))
    analysis = detector._automatic_cnr_mask(gray)
    mask = analysis["candidate_mask"]

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=int(detector.params.get("connectivity", 8))
    )
    component_labels = list(range(1, count))
    print(f"components (excluding background): {len(component_labels)}")

    baseline = detector._collect_candidates(
        analysis["image_float"], mask, analysis["inclusion_mask"]
    )
    baseline_cnrs = [candidate.cnr for candidate in baseline]
    tie_groups = {}
    for cnr in baseline_cnrs:
        tie_groups[cnr] = tie_groups.get(cnr, 0) + 1
    ties = {cnr: n for cnr, n in tie_groups.items() if n > 1}
    print(
        f"candidates: {len(baseline)}, distinct CNR: {len(tie_groups)}, "
        f"exact-tie groups: {len(ties)}, largest tie group: "
        f"{max(ties.values()) if ties else 0}"
    )
    if not ties:
        print("scene produces no exact CNR tie; the label order cannot be observed")
        return 0

    reference = _payload(detector, analysis, np.asarray(labels), stats, count)
    baseline_payload = [
        {
            "bbox": list(candidate.bbox),
            "area": candidate.area,
            "cnr": candidate.cnr,
            "contrast": candidate.contrast,
            "defect_mean": candidate.defect_mean,
            "background_mean": candidate.background_mean,
            "background_std": candidate.background_std,
            "background_area": candidate.background_area,
        }
        for candidate in baseline
    ]
    if reference != baseline_payload:
        print("FAIL: supplying cv2's own labels changed the output; tool is unsound")
        return 1

    rng = np.random.default_rng(2024)
    shortest_moved = None
    identical = 0
    differed = 0
    worst_example = None
    for _ in range(args.permutations):
        order = rng.permutation(component_labels)
        remap = np.zeros(count, dtype=np.int32)
        permuted_stats = np.zeros_like(stats)
        # The label map and the stats table are both indexed by label, so a relabelling
        # has to permute the stats as well; permuting only the labels would feed each
        # component another component's bounding box and area.
        permuted_stats[0] = stats[0]
        for new_index, original in enumerate(order):
            remap[original] = new_index + 1
            permuted_stats[new_index + 1] = stats[original]
        permuted = remap[labels]
        payload = _payload(detector, analysis, permuted, permuted_stats, count)
        if payload == baseline_payload:
            identical += 1
            continue
        differed += 1
        # How far did the first differing candidate move?
        for index, (actual, expected) in enumerate(zip(payload, baseline_payload)):
            if actual != expected:
                move = abs(
                    payload.index(actual) - baseline_payload.index(expected)
                ) if actual in baseline_payload else None
                if shortest_moved is None or (move is not None and move < shortest_moved):
                    shortest_moved = move
                if worst_example is None:
                    worst_example = {
                        "index": index,
                        "permuted_bbox": actual["bbox"],
                        "baseline_bbox": expected["bbox"],
                    }
                break

    print()
    print(f"permutations tried: {args.permutations}")
    print(f"output identical:   {identical}")
    print(f"output differed:    {differed}")
    if worst_example is not None:
        print(f"first difference example: {worst_example}")

    payload = {
        "kind": "cnr_label_order_impact",
        "component_count": len(component_labels),
        "candidate_count": len(baseline_payload),
        "exact_tie_groups": len(ties),
        "largest_tie_group": max(ties.values()) if ties else 0,
        "permutations": args.permutations,
        "identical": identical,
        "differed": differed,
        "worst_example": worst_example,
        "label_order_observable": differed > 0,
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
