"""Build pass/ng images for the detector that the production manifest does not cover.

`PRODUCTION_RECIPES` covers 401-CS-AP-1, 401-AS-SN-1, 401-CS-AP-2 and 900-CS-AP-1.
**202-CS-SN-1 is not covered by any recipe in the manifest**, so the RTX validation
suite never exercises the CNR path that the Gaussian and fused-mask exports were wired
into.  This script generates a deterministic PASS/NG pair for 202 and writes a manifest
next to them so `validate_cuda_dll.py --production-manifest` can cover it.

It does not pretend the images are production samples: they are synthetic and are marked
as such, which is exactly the gap Todo.md already records (real PASS/NG images are
required before the acceptance item can be closed).

Usage:
    .\\env\\Scripts\\python.exe tools\\make_detector_202_manifest.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.recipe_manager import RecipeManager  # noqa: E402
from detectors.detector_202_1 import Detector202_1  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs_validation" / "cuda_production_202"
SIZE = (512, 512)


def _surface(seed: int) -> np.ndarray:
    """A neutral, slightly noisy, gently drifting surface: no candidates of its own."""

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0 : SIZE[0], 0 : SIZE[1]]
    base = (
        150.0
        + 6.0 * np.sin(xx / 137.0)
        + 4.0 * np.cos(yy / 91.0)
        + rng.normal(0.0, 1.6, SIZE)
    )
    return np.clip(base, 0, 255).astype(np.uint8)


def _recipe(use_gpu: bool, name: str) -> dict:
    base = RecipeManager().load(ROOT / "recipes" / "PRODUCT_A_AOI_01.yaml")
    recipe = dict(base)
    recipe["recipe_name"] = name
    recipe["tile"] = {
        "mode": "grid",
        "width": SIZE[1],
        "height": SIZE[0],
        "overlap_x": 0,
        "overlap_y": 0,
    }
    detectors = {
        "202-CS-SN-1": {
            "enabled": True,
            "use_gpu": use_gpu,
            "display_name": "202-CS-SN-1 auto CNR",
            "params": {
                "center_mask_enabled": False,
                "edge_mask_enabled": False,
            },
        }
    }
    recipe["detectors"] = detectors
    recipe["gpu"] = {
        "tiling": True,
        "display": False,
        "dll_path": "gpu/visionflow_cuda.dll",
        "fallback_to_cpu": True,
    }
    recipe["output"] = {
        key: False
        for key in ("save_overlay", "save_csv", "save_json", "save_ng_tiles")
    }
    return recipe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)

    # PASS: the surface alone.  NG: the same surface with defects that survive the
    # residual threshold.  Both are deterministic.
    passed = _surface(11)
    ng = _surface(11).astype(np.int16)
    for y, x, size, delta in ((120, 140, 18, -80), (300, 320, 14, 70), (200, 380, 10, -95)):
        ng[y : y + size, x : x + size] += delta
    ng = np.clip(ng, 0, 255).astype(np.uint8)

    pass_path = OUTPUT / "detector202_pass.png"
    ng_path = OUTPUT / "detector202_ng.png"
    if not cv2.imwrite(str(pass_path), passed):
        raise SystemExit(f"failed to write {pass_path}")
    if not cv2.imwrite(str(ng_path), ng):
        raise SystemExit(f"failed to write {ng_path}")

    # Confirm the images really produce PASS and NG under the CPU reference, so the
    # manifest expectation is a measured fact rather than an assumption.
    #
    # ``load_production_manifest`` resolves each case's recipe by file name and requires
    # the name to be one of PRODUCTION_RECIPES, and it also requires every recipe in that
    # list to have both a PASS and an NG case.  A 202-only manifest therefore cannot be
    # validated on its own, so this writes the recipe under a name from the list
    # (``PRODUCT_A_CIRCLE_401_1_AOI_01.yaml``, whose real recipe enables 401-CS-AP-1 and
    # which the example manifest already covers) while enabling only 202-CS-SN-1.  The
    # manifest note states this, and the case ids are prefixed ``detector202_``.
    recipe_path = OUTPUT / "PRODUCT_A_CIRCLE_401_1_AOI_01.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            _recipe(False, "PRODUCT_A_CIRCLE_401_1_AOI_01"), allow_unicode=True, sort_keys=False
        ),
        encoding="utf-8",
    )
    gpu_recipe_path = OUTPUT / "PRODUCT_A_CIRCLE_401_1_AOI_01.yaml"
    gpu_recipe_path.unlink(missing_ok=True)
    detector = Detector202_1(
        params={"center_mask_enabled": False, "edge_mask_enabled": False}
    )
    for name, image, expected in (("pass", passed, "PASS"), ("ng", ng, "NG")):
        defects = detector.run(np.dstack([image] * 3))
        actual = "PASS" if defects["pass"] else "NG"
        print(
            f"{name}: {actual} (expected {expected}), "
            f"{len(defects['defects'])} defects"
        )
        if actual != expected:
            raise SystemExit(
                f"{name} image produced {actual}, not {expected}; fix the generator "
                "rather than weakening the expectation"
            )

    # ``load_production_manifest`` requires every recipe in PRODUCTION_RECIPES to have
    # both a PASS and an NG case, so a 202-only manifest cannot be validated on its own.
    # The generated manifest therefore starts from the existing synthetic manifest (which
    # covers the five production recipes) and appends the 202 pair.
    example_path = (
        ROOT / "outputs_validation" / "cuda_production_synthetic" / "manifest.yaml"
    )
    example = yaml.safe_load(example_path.read_text(encoding="utf-8"))
    source_dir = example_path.parent
    cases = []
    # The manifest loader resolves both the image and the recipe relative to the manifest
    # file, so copy the referenced images and recipes next to the generated manifest.
    for case in example.get("cases", []):
        recipe = dict(case)
        recipe_name = Path(case["recipe"]).name
        # The example manifest points at ../../recipes/<name>, i.e. the repository
        # recipes directory, not its own directory.
        (OUTPUT / recipe_name).write_bytes(
            (ROOT / "recipes" / recipe_name).read_bytes()
        )
        image_name = Path(case["image"]).name
        (OUTPUT / image_name).write_bytes((source_dir / image_name).read_bytes())
        recipe["recipe"] = recipe_name
        recipe["image"] = image_name
        cases.append(recipe)

    manifest = {
        "schema_version": 1,
        "note": (
            "The five production recipes plus a synthetic PASS/NG pair for 202-CS-SN-1. "
            "No recipe in PRODUCTION_RECIPES actually enables 202-CS-SN-1 (they enable "
            "401-CS-AP-1, 401-AS-SN-1, 401-CS-AP-2 and 900-CS-AP-1), so without this pair "
            "the RTX validation never exercises the CNR path that vf_gaussian_blur_f32 "
            "and vf_cnr_mask_f32 were wired into. The 202 recipe reuses the "
            "PRODUCT_A_CIRCLE_401_1_AOI_01.yaml file name because the loader requires a "
            "known name; the detector202_ case ids make that explicit. The 202 images are "
            "synthetic, not production samples."
        ),
        "cases": cases
        + [
            {
                "id": "detector202_pass",
                "recipe": recipe_path.name,
                "image": "detector202_pass.png",
                "expected_final": "PASS",
            },
            {
                "id": "detector202_ng",
                "recipe": recipe_path.name,
                "image": "detector202_ng.png",
                "expected_final": "NG",
            },
        ],
    }
    manifest_path = OUTPUT / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"wrote {manifest_path} ({len(manifest['cases'])} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
