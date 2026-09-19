"""How much of a real-scene defect list depends on the CCL label numbering?

`Detector202_1._collect_candidates` walks components in ascending label order and then
sorts by CNR with Python's stable sort, so candidates whose CNR is *exactly* equal stay
in label order.  Only those candidates can be reordered by a replacement
connected-components implementation that numbers components differently.

This tool measures, on production-shaped ROIs, how many candidates sit in an exact CNR
tie group.  That number is the share of the defect list whose order a GPU CCL would have
to reproduce, and it decides how much the label-ordering gap actually costs.

Usage:
    .\\env\\Scripts\\python.exe tools\\cnr_tie_share.py [--json OUTPUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detectors.detector_202_1 import Detector202_1  # noqa: E402

PARAMS = {"center_mask_enabled": False, "edge_mask_enabled": False}


def _scene(height: int, width: int, defects: int, seed: int = 2021) -> np.ndarray:
    """The same generator the production benchmark uses, so the numbers are comparable."""

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    base = (
        148.0
        + 7.0 * np.sin(xx / 311.0)
        + 5.0 * np.cos(yy / 197.0)
        + rng.normal(0.0, 2.2, (height, width))
    )
    gray = base
    for _ in range(defects):
        y = int(rng.integers(60, height - 60))
        x = int(rng.integers(60, width - 60))
        size = int(rng.integers(9, 26))
        gray[y : y + size, x : x + size] += float(rng.choice([-1.0, 1.0])) * float(
            rng.uniform(22.0, 85.0)
        )
    return np.clip(gray, 0, 255).astype(np.uint8)


def _regular_array(height: int, width: int, pitch_y: int, pitch_x: int) -> np.ndarray:
    gray = np.full((height, width), 150, dtype=np.uint8)
    for y in range(20, height - 20, pitch_y):
        for x in range(20, width - 20, pitch_x):
            gray[y : y + 10, x : x + 10] = 60
    return gray


def _measure(name: str, gray: np.ndarray) -> dict:
    detector = Detector202_1(params=dict(PARAMS))
    analysis = detector._automatic_cnr_mask(gray)
    candidates = detector._collect_candidates(
        analysis["image_float"],
        analysis["candidate_mask"],
        analysis["inclusion_mask"],
    )
    counts: dict[float, int] = {}
    for candidate in candidates:
        counts[candidate.cnr] = counts.get(candidate.cnr, 0) + 1
    tie_groups = {cnr: n for cnr, n in counts.items() if n > 1}
    tied = sum(tie_groups.values())
    return {
        "scene": name,
        "shape": list(gray.shape),
        "candidates": len(candidates),
        "tie_groups": len(tie_groups),
        "tied_candidates": tied,
        "largest_tie_group": max(tie_groups.values()) if tie_groups else 0,
        "tied_share": tied / len(candidates) if candidates else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    rows = [
        _measure("prod-2000x12000-defects96", _scene(2000, 12000, 96)),
        _measure("prod-2000x4000-defects24", _scene(2000, 4000, 24)),
        _measure("prod-2000x12000-clean", _scene(2000, 12000, 0)),
        _measure("regular-array-pitch40", _regular_array(640, 800, 40, 40)),
        _measure("regular-array-pitch64", _regular_array(640, 800, 64, 64)),
        _measure("regular-array-pitch128", _regular_array(640, 800, 128, 128)),
    ]

    for row in rows:
        print(
            f"{row['scene']:30s} candidates={row['candidates']:5d} "
            f"tie_groups={row['tie_groups']:4d} tied={row['tied_candidates']:5d} "
            f"({row['tied_share'] * 100:5.1f}%) largest={row['largest_tie_group']}"
        )

    production = [row for row in rows if row["scene"].startswith("prod-")]
    arrays = [row for row in rows if row["scene"].startswith("regular")]
    print()
    print(
        "production-shaped noisy scenes: "
        f"{sum(row['tied_candidates'] for row in production)} of "
        f"{sum(row['candidates'] for row in production)} candidates in a tie group"
    )
    print(
        "regular part arrays: "
        f"{sum(row['tied_candidates'] for row in arrays)} of "
        f"{sum(row['candidates'] for row in arrays)} candidates in a tie group"
    )
    print(
        "interpretation: only tied candidates can be reordered by a differently numbered "
        "CCL. A regular array of identical parts is the worst case; a noisy production "
        "surface is the best case."
    )

    payload = {
        "kind": "cnr_tie_share",
        "rows": rows,
        "production_tied": sum(row["tied_candidates"] for row in production),
        "production_candidates": sum(row["candidates"] for row in production),
        "array_tied": sum(row["tied_candidates"] for row in arrays),
        "array_candidates": sum(row["candidates"] for row in arrays),
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
