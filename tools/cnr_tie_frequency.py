"""How often do realistic scenes produce an exact CNR tie in 202-CS-SN-1?

An exact tie is the only way the component label order becomes observable in the
detector output (see ``tools/cnr_label_order_impact.py``).  Symmetric synthetic grids
produce hundreds of them, so the question that decides the real severity of the
connected-components label-order gap is whether *realistic* scenes do too.

The sweep covers the two production ROI shapes plus assorted content: sensor-like
noise, illumination gradients, clustered defects, regular part arrays and everything
in between.  Every component of every scene is checked for an exactly equal CNR, and
the tool reports how many individual components belong to a tie group.

Usage:
    .\\env\\Scripts\\python.exe tools\\cnr_tie_frequency.py [--json OUTPUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detectors.detector_202_1 import Detector202_1  # noqa: E402
from tools.benchmark_median_202 import PARAMS, _scene  # noqa: E402


def _realistic_scenes():
    rng = np.random.default_rng(20240915)

    # 1. production-shaped ROI with embedded defects at several densities
    for defects in (0, 1, 6, 24, 96):
        yield f"prod-2000x4000-defects{defects}", _scene(2000, 4000, defects)
    yield "prod-2000x12000-defects96", _scene(2000, 12000, 96)

    # 2. constant background with sensor noise only (no defects at all)
    for noise in (0.5, 1.0, 2.5, 6.0):
        base = 150.0 + rng.normal(0.0, noise, (600, 800))
        yield f"noise{noise}-no-defects", np.clip(base, 0, 255).astype(np.uint8)

    # 3. a regular product array: the realistic version of the symmetric grid
    for pitch in (40, 64, 128):
        gray = np.full((640, 800), 150, dtype=np.uint8)
        for y in range(30, 640 - 30, pitch):
            for x in range(30, 800 - 30, pitch):
                gray[y : y + 12, x : x + 12] = 60
        yield f"part-array-pitch{pitch}", gray

    # 4. clustered defects on a textured background
    for seed in range(6):
        local = np.random.default_rng(1000 + seed)
        yy, xx = np.mgrid[0:512, 0:512]
        gray = (
            145.0
            + 12.0 * np.sin(xx / 53.0)
            + 8.0 * np.cos(yy / 37.0)
            + local.normal(0.0, 3.0, (512, 512))
        )
        for _ in range(9):
            y = int(local.integers(30, 470))
            x = int(local.integers(30, 470))
            size = int(local.integers(6, 20))
            gray[y : y + size, x : x + size] += float(local.choice([-1, 1])) * float(
                local.uniform(15.0, 70.0)
            )
        yield f"textured-cluster-seed{seed}", np.clip(gray, 0, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    scenes_with_ties = 0
    total_components = 0
    total_tied_components = 0
    for name, gray in _realistic_scenes():
        detector = Detector202_1(params=dict(PARAMS))
        analysis = detector._automatic_cnr_mask(gray)
        candidates = detector._collect_candidates(
            analysis["image_float"],
            analysis["candidate_mask"],
            analysis["inclusion_mask"],
        )
        groups: dict[float, int] = {}
        for candidate in candidates:
            groups[candidate.cnr] = groups.get(candidate.cnr, 0) + 1
        ties = {cnr: count for cnr, count in groups.items() if count > 1}
        tied_components = sum(ties.values())
        scenes_with_ties += 1 if ties else 0
        total_components += len(candidates)
        total_tied_components += tied_components
        rows.append(
            {
                "scene": name,
                "shape": list(gray.shape),
                "candidate_count": len(candidates),
                "distinct_cnr": len(groups),
                "tie_groups": len(ties),
                "tied_components": tied_components,
                "largest_tie_group": max(ties.values()) if ties else 0,
            }
        )
        marker = "" if not ties else "   <-- exact CNR tie"
        print(
            f"{name:28s} candidates={len(candidates):4d} "
            f"distinct={len(groups):4d} tie_groups={len(ties):3d} "
            f"tied={tied_components:4d}{marker}"
        )

    print()
    print(f"scenes: {len(rows)}")
    print(f"scenes with at least one exact CNR tie: {scenes_with_ties}/{len(rows)}")
    print(f"components: {total_components}, of which tied: {total_tied_components}")
    print(
        "interpretation: a tie makes the component label order observable in the "
        "defect list order, so every tied component is a scene where a "
        "differently-numbered CCL would reorder the output"
    )

    payload = {
        "kind": "cnr_tie_frequency",
        "scene_count": len(rows),
        "scenes_with_ties": scenes_with_ties,
        "components": total_components,
        "tied_components": total_tied_components,
        "scenes": rows,
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
