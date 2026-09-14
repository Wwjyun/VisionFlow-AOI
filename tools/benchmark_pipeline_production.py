"""End-to-end production-shape validation of the whole pipeline, CPU vs CUDA.

The RTX 3090 evidence gathered so far is either per-operator (median, Gaussian,
anchor) or a single synthetic ROI.  This tool exercises the **whole pipeline** -
recipe load, whole-image H2D, tiling, detector, aggregation, reporting - at the
production 202-CS-SN-1 shape: a large canvas carrying six 2000x12000 ROIs, which is
the configuration the Todo records as the CPU bottleneck (CPU detector ~4.9 s).

It reports, for one image and one recipe:

* the normalised final result of the CPU run and the CUDA run, compared field by
  field (PASS/NG, defect count, bbox, area, confidence, metadata, tile order),
* stage timings for both backends,
* ``execution.gpu.device_host_split`` and the resident-upload record, so the
  actual device/host split is reported from the run rather than asserted.

Usage:
    .\\env\\Scripts\\python.exe tools\\benchmark_pipeline_production.py [--json OUT]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pipeline import AOIPipeline  # noqa: E402
from core.recipe_manager import RecipeManager  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

CANVAS = (4000, 6000)  # (width, height) = 2 ROI widths x 3 ROI heights
ROI = (2000, 2000)  # (height, width)
# Production geometry recorded in Todo.md: six 2000x12000 ROIs (height x width), which
# is the shape the CPU detector was measured at ~4.9 s.  Two columns of width 12000 and
# three rows of height 2000 need a 24000x6000 canvas, i.e. 432 MB as BGR, so it is
# opt-in via ``--profile production`` while the quick 2000x2000 geometry is the default.
# Note the shape matters: 2000x12000 and 12000x2000 have the same pixel count but very
# different cache behaviour for a 51-pixel separable filter, so the orientation is part
# of the measurement rather than an implementation detail.
PRODUCTION_CANVAS = (24000, 6000)
PRODUCTION_ROI = (2000, 12000)


def _roi_origins(roi: tuple[int, int], columns: int, rows: int):
    return [(row * roi[0], col * roi[1]) for row in range(rows) for col in range(columns)]


# Exactly six ROIs: two columns of width 2000, three rows of height 2000.  ``tile.mode:
# grid`` with width=2000/height=2000 and no overlap then produces exactly these six
# tiles, so the pipeline performs one whole-image upload while the detector sees six
# tiles.  No real production image is checked in (see the acceptance items in Todo.md),
# so the tool validates the *pipeline* path and the CPU/CUDA agreement at production-like
# sizes.
ROI_ORIGINS = _roi_origins(ROI, columns=2, rows=3)
PRODUCTION_ROI_ORIGINS = _roi_origins(PRODUCTION_ROI, columns=2, rows=3)

DETECTOR_ID = "202-CS-SN-1"
DETECTOR_PARAMS = {
    "center_mask_enabled": False,
    "edge_mask_enabled": False,
    "min_component_area_px": 5,
    "min_component_area_ratio": 0.000001,
    "max_component_area_ratio": 0.05,
}


def _render_canvas(
    canvas_path: Path,
    seed: int = 202,
    canvas: tuple[int, int] = CANVAS,
    roi: tuple[int, int] = ROI,
    roi_origins: list[tuple[int, int]] = ROI_ORIGINS,
) -> None:
    """Write the synthetic canvas without holding more than one plane at a time."""

    width, height = canvas
    rng = np.random.default_rng(seed)
    bgr = np.empty((height, width, 3), np.uint8)

    # Background: a neutral, slightly noisy surface outside the ROI slots.
    bgr[:] = 128
    for start in range(0, height, 2048):
        stop = min(height, start + 2048)
        noise = rng.normal(0.0, 1.0, (stop - start, width, 1))
        bgr[start:stop] = np.clip(128.0 + noise, 0, 255).astype(np.uint8)

    # Each ROI slot gets a regular part array with a handful of defects, which is the
    # content that produces exact CNR ties.
    for index, (y0, x0) in enumerate(roi_origins):
        roi_height, roi_width = roi[0], roi[1]
        yy, xx = np.mgrid[0:roi_height, 0:roi_width]
        base = (
            150.0
            + 8.0 * np.sin(xx / 311.0)
            + 5.0 * np.cos(yy / 197.0)
            + rng.normal(0.0, 2.0, (roi_height, roi_width))
        )
        local = np.clip(base, 0, 255)
        # A grid of parts covering the whole ROI, then a few defects on top.
        pitch_y, pitch_x = 400, 600
        for py in range(120, roi_height - 120, pitch_y):
            for px in range(120, roi_width - 120, pitch_x):
                local[py : py + 14, px : px + 14] = 60.0
        local_rng = np.random.default_rng(seed + 100 + index)
        for _ in range(3):
            dy = int(local_rng.integers(150, roi_height - 150))
            dx = int(local_rng.integers(150, roi_width - 150))
            size = int(local_rng.integers(10, 22))
            local[dy : dy + size, dx : dx + size] += float(
                local_rng.choice([-1.0, 1.0])
            ) * float(local_rng.uniform(30.0, 80.0))
        local = np.clip(local, 0, 255).astype(np.uint8)
        bgr[y0 : y0 + roi_height, x0 : x0 + roi_width, 0] = local
        bgr[y0 : y0 + roi_height, x0 : x0 + roi_width, 1] = local
        bgr[y0 : y0 + roi_height, x0 : x0 + roi_width, 2] = local

    canvas_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(canvas_path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise SystemExit(f"failed to write {canvas_path}")
    del bgr


def _recipe(
    base: dict, dll_path: str, use_gpu: bool, roi: tuple[int, int] = ROI
) -> dict:
    recipe = copy.deepcopy(base)
    recipe["gpu"] = {
        "tiling": use_gpu,
        "display": use_gpu,
        "dll_path": dll_path,
        "fallback_to_cpu": False,
        "mode": "cuda" if use_gpu else "cpu",
    }
    detectors = recipe.setdefault("detectors", {})
    for detector_id in list(detectors):
        detectors[detector_id]["enabled"] = detector_id == DETECTOR_ID
        detectors[detector_id]["use_gpu"] = False
    detectors[DETECTOR_ID] = {
        "enabled": True,
        "use_gpu": use_gpu,
        "display_name": "202-CS-SN-1 auto CNR (production shape)",
        "params": dict(DETECTOR_PARAMS),
    }
    # A grid whose cell is exactly one ROI yields exactly the intended tiles.
    recipe["tile"] = {
        "mode": "grid",
        "width": roi[1],
        "height": roi[0],
        "overlap_x": 0,
        "overlap_y": 0,
    }
    recipe["output"] = {
        key: False
        for key in ("save_overlay", "save_csv", "save_json", "save_ng_tiles")
    }
    return recipe


def _normalised(result: dict) -> dict:
    normalised = copy.deepcopy(result)
    for key in ("duration_sec", "outputs", "execution"):
        normalised.pop(key, None)
    provenance = normalised.get("provenance")
    if isinstance(provenance, dict):
        provenance.pop("recipe_source_sha256", None)
        provenance.pop("effective_recipe_sha256", None)
    for tile_result in normalised.get("tiles", []):
        for detector_result in tile_result.get("detectors", []):
            detector_result.pop("execution", None)
    return normalised


# The device float32 Gaussian is mathematically equivalent, not bit-identical, so four
# residual-derived diagnostics drift in their last bits.  They are reported separately
# instead of being hidden, and the decision-bearing fields are compared strictly.
_DRIFTING_METADATA = {
    "mad",
    "residual_median",
    "residual_threshold",
    "robust_noise_sigma",
}

# Fields that exist precisely to say *which* backend produced the record.  They must
# differ between the CPU and CUDA runs, so comparing them as decision-bearing would
# report a difference on every defect and hide a real mismatch.
_BACKEND_PROVENANCE = {
    "background_backend",
    "residual_backend",
    "background_precision_note",
}


def _split_defects(result: dict):
    """Return ``(decision_fields, drifting_fields, worst_drift)`` for one result."""

    decision = []
    drifting = []
    worst = 0.0
    for tile_result in result.get("tiles", []):
        for detector_result in tile_result.get("detectors", []):
            decision.append(
                (
                    tile_result.get("tile", {}).get("tile_id"),
                    detector_result.get("detector_id"),
                    detector_result.get("pass"),
                    detector_result.get("defect_count"),
                )
            )
            for defect in detector_result.get("defects", []):
                metadata = defect.get("metadata", {}) or {}
                decision.append(
                    (
                        defect.get("type"),
                        tuple(defect.get("bbox_local") or ()),
                        defect.get("area"),
                        defect.get("confidence"),
                    )
                )
                for name, value in sorted(metadata.items()):
                    if name in _DRIFTING_METADATA and isinstance(value, (int, float)):
                        drifting.append((name, float(value)))
                    elif name in _BACKEND_PROVENANCE:
                        continue
                    else:
                        decision.append((name, value))
    return decision, drifting, worst


def _compare(cpu_result: dict, gpu_result: dict) -> dict:
    cpu_decision, cpu_drift, _ = _split_defects(cpu_result)
    gpu_decision, gpu_drift, _ = _split_defects(gpu_result)
    decision_equal = cpu_decision == gpu_decision
    drift_counts = {}
    worst_drift = 0.0
    for (name, cpu_value), (gpu_name, gpu_value) in zip(cpu_drift, gpu_drift):
        if name != gpu_name:
            decision_equal = False
            break
        delta = abs(cpu_value - gpu_value)
        worst_drift = max(worst_drift, delta)
        if delta > 0.0:
            drift_counts[name] = drift_counts.get(name, 0) + 1
    return {
        "decision_equal": decision_equal,
        "drift_counts": drift_counts,
        "worst_drift": worst_drift,
        "drifting_field_count": len(cpu_drift),
    }


def _run(recipe_path: Path, image: Path, output_dir: Path):
    started = time.perf_counter()
    result = AOIPipeline(recipe_path, output_dir).run(image)
    return (time.perf_counter() - started) * 1000.0, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", default=str(ROOT / "gpu" / "visionflow_cuda.dll"))
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--keep", action="store_true", help="keep the generated canvas")
    parser.add_argument("--work", type=Path, default=None)
    parser.add_argument(
        "--profile",
        choices=("quick", "production"),
        default="quick",
        help=(
            "quick: six 2000x2000 ROIs on a 4000x6000 canvas; "
            "production: six 2000x12000 ROIs on a 6000x24000 canvas (the geometry "
            "recorded in Todo.md), which needs ~432 MB for the canvas"
        ),
    )
    args = parser.parse_args()

    work = args.work or Path(tempfile.mkdtemp(prefix="visionflow_production_"))
    canvas = work / "production_canvas.png"
    if args.profile == "production":
        canvas_size, roi, roi_origins = (
            PRODUCTION_CANVAS,
            PRODUCTION_ROI,
            PRODUCTION_ROI_ORIGINS,
        )
    else:
        canvas_size, roi, roi_origins = CANVAS, ROI, ROI_ORIGINS
    print(
        f"profile: {args.profile}  canvas {canvas_size[0]}x{canvas_size[1]} (w x h), "
        f"{len(roi_origins)} ROIs {roi[0]}h x {roi[1]}w"
    )
    if not canvas.is_file() or not args.keep:
        started = time.perf_counter()
        _render_canvas(
            canvas, canvas=canvas_size, roi=roi, roi_origins=roi_origins
        )
        print(
            f"wrote {canvas} ({canvas.stat().st_size / 1e6:.1f} MB) in "
            f"{time.perf_counter() - started:.1f} s"
        )

    base = RecipeManager().load(ROOT / "recipes" / "PRODUCT_A_AOI_01.yaml")
    cpu_recipe = _recipe(base, args.dll, use_gpu=False, roi=roi)
    gpu_recipe = _recipe(base, args.dll, use_gpu=True, roi=roi)
    cpu_path = work / "cpu.yaml"
    gpu_path = work / "gpu.yaml"
    cpu_path.write_text(yaml.safe_dump(cpu_recipe, allow_unicode=True, sort_keys=False), encoding="utf-8")
    gpu_path.write_text(yaml.safe_dump(gpu_recipe, allow_unicode=True, sort_keys=False), encoding="utf-8")

    cpu_ms, cpu_result = _run(cpu_path, canvas, work / "cpu_out")
    gpu_ms, gpu_result = _run(gpu_path, canvas, work / "gpu_out")

    cpu_final = cpu_result.get("final_result")
    gpu_final = gpu_result.get("final_result")
    cpu_summary = cpu_result.get("summary", {})
    gpu_summary = gpu_result.get("summary", {})
    print()
    print(f"CPU  end-to-end {cpu_ms:9.1f} ms  final={cpu_final}  {cpu_summary}")
    print(f"CUDA end-to-end {gpu_ms:9.1f} ms  final={gpu_final}  {gpu_summary}")
    print(f"speedup {cpu_ms / gpu_ms:.2f}x")

    cpu_normalised = _normalised(cpu_result)
    gpu_normalised = _normalised(gpu_result)
    strictly_identical = cpu_normalised == gpu_normalised
    comparison = _compare(cpu_result, gpu_result)
    print(f"normalised results strictly identical: {strictly_identical}")
    print(
        "decision-bearing fields identical (PASS/NG, defect count, type, bbox, area, "
        f"confidence, remaining metadata): {comparison['decision_equal']}"
    )
    if comparison["drift_counts"]:
        print(
            f"residual-derived diagnostics that drifted: {comparison['drift_counts']} "
            f"over {comparison['drifting_field_count']} values, "
            f"worst |drift| {comparison['worst_drift']:.3e}"
        )
    else:
        print("residual-derived diagnostics: no drift on this image")

    execution = gpu_result.get("execution", {}).get("gpu", {})
    split = execution.get("device_host_split")
    resident = execution.get("resident_image")
    tiling = execution.get("tiling")
    print()
    print("device/host split reported by the run:")
    if isinstance(split, dict):
        for stage, side in split.items():
            print(f"  {stage:32s} {side}")
    else:
        print(f"  {split!r}")
    print(f"resident_image: {resident}")
    if isinstance(tiling, dict):
        print(
            f"tiling: active={tiling.get('active')} tiles={tiling.get('tile_count')} "
            f"backend={tiling.get('backend')}"
        )
    if not comparison["decision_equal"]:
        print()
        print("DECISION-BEARING DIFFERENCE - the CUDA run is not equivalent:")
        cpu_tiles = cpu_normalised.get("tiles", [])
        gpu_tiles = gpu_normalised.get("tiles", [])
        for index, (cpu_tile, gpu_tile) in enumerate(zip(cpu_tiles, gpu_tiles)):
            if cpu_tile != gpu_tile:
                print(f"  tile {index}: CPU={json.dumps(cpu_tile, default=str)[:400]}")
                print(f"  tile {index}: GPU={json.dumps(gpu_tile, default=str)[:400]}")
                break

    # Pipeline stage timings live under execution.performance: stages_sec holds the
    # pipeline-level stages and detector_stages_sec holds each detector's own stages.
    cpu_stage = dict(
        cpu_result.get("execution", {}).get("performance", {}).get("stages_sec", {})
    )
    gpu_stage = dict(
        gpu_result.get("execution", {}).get("performance", {}).get("stages_sec", {})
    )
    cpu_detector_stage = {
        f"{detector_id}.{stage}": seconds
        for detector_id, stages in cpu_result.get("execution", {})
        .get("performance", {})
        .get("detector_stages_sec", {})
        .items()
        for stage, seconds in stages.items()
    }
    gpu_detector_stage = {
        f"{detector_id}.{stage}": seconds
        for detector_id, stages in gpu_result.get("execution", {})
        .get("performance", {})
        .get("detector_stages_sec", {})
        .items()
        for stage, seconds in stages.items()
    }
    if cpu_detector_stage or gpu_detector_stage:
        print()
        print(f"{'detector stage':40s} {'CPU ms':>10s} {'CUDA ms':>10s}")
        for key in sorted(set(cpu_detector_stage) | set(gpu_detector_stage)):
            print(
                f"{key:40s} {cpu_detector_stage.get(key, float('nan')) * 1000.0:10.2f} "
                f"{gpu_detector_stage.get(key, float('nan')) * 1000.0:10.2f}"
            )
    if cpu_stage and gpu_stage:
        print()
        print(f"{'pipeline stage':40s} {'CPU ms':>10s} {'CUDA ms':>10s}")
        for key in sorted(set(cpu_stage) | set(gpu_stage)):
            print(
                f"{key:40s} {cpu_stage.get(key, float('nan')) * 1000.0:10.2f} "
                f"{gpu_stage.get(key, float('nan')) * 1000.0:10.2f}"
            )

    payload = {
        "kind": "pipeline_production",
        "profile": args.profile,
        "canvas": {"width": canvas_size[0], "height": canvas_size[1]},
        "roi": {"height": roi[0], "width": roi[1], "origins": roi_origins},
        "cpu_ms": cpu_ms,
        "gpu_ms": gpu_ms,
        "speedup": cpu_ms / gpu_ms if gpu_ms else None,
        "cpu_final": cpu_final,
        "gpu_final": gpu_final,
        "cpu_summary": cpu_summary,
        "gpu_summary": gpu_summary,
        "normalised_identical": strictly_identical,
        "decision_fields_identical": comparison["decision_equal"],
        "drifting_diagnostics": comparison["drift_counts"],
        "worst_diagnostic_drift": comparison["worst_drift"],
        "device_host_split": split,
        "resident_image": resident,
        "tiling": tiling,
        "cpu_stage_ms": {key: value * 1000.0 for key, value in cpu_stage.items()},
        "gpu_stage_ms": {key: value * 1000.0 for key, value in gpu_stage.items()},
        "cpu_detector_stage_ms": {
            key: value * 1000.0 for key, value in cpu_detector_stage.items()
        },
        "gpu_detector_stage_ms": {
            key: value * 1000.0 for key, value in gpu_detector_stage.items()
        },
        "gpu_metrics": gpu_result.get("execution", {}).get("gpu", {}).get("metrics", {}),
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0 if comparison["decision_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
