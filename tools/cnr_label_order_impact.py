"""Does the component label order actually change 202-CS-SN-1's final output?

``Detector202_1._collect_candidates`` walks components in ascending label order and
then sorts the candidates by CNR with Python's **stable** sort.  A stable sort keeps
the label visit order for candidates whose CNR is *exactly* equal, so in principle a
replacement connected-components implementation that numbers components differently
could reorder the defect list.  Exact CNR ties do occur: a symmetric grid of identical
defects on a constant background produces many.

This tool settles the question empirically instead of arguing from the contract:

* build a scene that produces exact CNR ties,
* relabel the component map with an arbitrary permutation of the component numbers
  (which is exactly what a differently-numbered CCL implementation would hand over),
* re-run ``_collect_candidates`` on the permuted labels and compare the resulting
  defect list, field by field, with the original.

If the output is unchanged for every permutation then the label order is not
observable in the detector output, and a GPU CCL only has to reproduce the component
*set* and the per-component stats, not OpenCV's numbering.

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
        for new_index, original in enumerate(order):
            remap[original] = new_index + 1
        permuted = remap[labels]
        payload = _payload(detector, analysis, permuted, stats, count)
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
