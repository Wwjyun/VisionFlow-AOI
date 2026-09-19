"""Measure how far OpenCV's 8-connectivity numbering departs from raster order.

This bounds the connected-components label-order gap.  The reference implementation's
numbering rule is already OpenCV's (`flattenL` numbers a component by the rank of its
smallest provisional label), so the remaining error is entirely the order in which the
Bolelli 2x2 block scan creates provisional labels.  Knowing *how far* that order departs
from pixel-raster order tells a future port how much is at stake and gives a cheap
sanity bound: an implementation that is off by more than the measured maximum is wrong
for a different reason.

The measurement is: for every component, take its raster-first pixel and rank all the
components by that pixel in raster order; then compare that rank with OpenCV's label.

Usage:
    .\\env\\Scripts\\python.exe tools\\cc_label_order_bound.py [--json OUTPUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _measure(mask: np.ndarray, connectivity: int) -> dict:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=connectivity
    )
    positions = []
    for label in range(1, count):
        ys, xs = np.nonzero(labels == label)
        order = np.lexsort((xs, ys))
        positions.append((int(ys[order[0]]), int(xs[order[0]]), label))
    ranked = sorted(positions)
    deltas = [
        label - (index + 1) for index, (_, _, label) in enumerate(ranked)
    ]
    return {
        "components": count - 1,
        "deltas": deltas,
        "nonzero": sum(1 for delta in deltas if delta),
        "min": min(deltas) if deltas else 0,
        "max": max(deltas) if deltas else 0,
        "max_abs": max((abs(delta) for delta in deltas), default=0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    shapes = [
        (1, 64),
        (64, 1),
        (3, 3),
        (8, 8),
        (17, 23),
        (32, 40),
        (33, 41),
        (64, 80),
        (129, 257),
        (400, 600),
    ]
    rows = []
    for shape in shapes:
        for density in (0.2, 0.45, 0.7):
            mask = (
                np.random.default_rng(0).random(shape) < density
            ).astype(np.uint8) * 255
            for connectivity in (8, 4):
                report = _measure(mask, connectivity)
                rows.append(
                    {
                        "shape": list(shape),
                        "density": density,
                        "connectivity": connectivity,
                        "components": report["components"],
                        "nonzero": report["nonzero"],
                        "min_delta": report["min"],
                        "max_delta": report["max"],
                        "max_abs_delta": report["max_abs"],
                    }
                )
                label = f"{shape[0]}x{shape[1]} c{connectivity} d{density}"
                print(
                    f"{label:22s} components={report['components']:5d} "
                    f"out-of-order={report['nonzero']:5d} "
                    f"delta in [{report['min']:3d}, {report['max']:3d}]"
                )

    eight = [row for row in rows if row["connectivity"] == 8]
    four = [row for row in rows if row["connectivity"] == 4]
    print()
    print(
        "4-connectivity: max |delta| over all cases = "
        f"{max(row['max_abs_delta'] for row in four)} "
        "(0 means the numbering is exactly raster order)"
    )
    print(
        "8-connectivity: max |delta| over all cases = "
        f"{max(row['max_abs_delta'] for row in eight)}"
    )
    print(
        "interpretation: OpenCV's 8-connectivity numbering is raster order perturbed by "
        "at most this many positions, so the label-order gap is bounded rather than "
        "arbitrary"
    )

    payload = {
        "kind": "cc_label_order_bound",
        "four_connectivity_max_abs_delta": max(
            row["max_abs_delta"] for row in four
        ),
        "eight_connectivity_max_abs_delta": max(
            row["max_abs_delta"] for row in eight
        ),
        "rows": rows,
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
