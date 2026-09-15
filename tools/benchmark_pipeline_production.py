"""End-to-end production-shape validation of the whole pipeline, CPU vs CUDA.

The RTX 3090 evidence gathered so far is either per-operator (median, Gaussian,
anchor) or a single synthetic ROI.  This tool exercises the **whole pipeline** -
recipe load, whole-image H2D, tiling, detector, aggregation, reporting - at the
production 202-CS-SN-1 shape: a 16384x13000 canvas carrying six 12000h x 2000w ROIs, which is
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
import math
import statistics
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pipeline import AOIPipeline  # noqa: E402
from core.gpu_session import GpuExecutionSession  # noqa: E402
from core.recipe_manager import RecipeManager  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

CANVAS = (4000, 6000)  # (width, height) = 2 ROI widths x 3 ROI heights
ROI = (2000, 2000)  # (height, width)
# User-confirmed production geometry: the source image is 16384w x 13000h and contains six
# 12000h x 2000w inspection ROIs. A small synthetic anchor selects one row of six ROIs so the
# benchmark exercises the same whole-image upload and device ROI path as production.
PRODUCTION_CANVAS = (16384, 13000)
PRODUCTION_ROI = (12000, 2000)
PRODUCTION_ANCHOR = (128, 128)  # x, y
PRODUCTION_BASE = (500, 500)  # x, y of the first ROI
PRODUCTION_GAP_X = 100


def _roi_origins(roi: tuple[int, int], columns: int, rows: int):
    return [(row * roi[0], col * roi[1]) for row in range(rows) for col in range(columns)]


# Exactly six ROIs: two columns of width 2000, three rows of height 2000.  ``tile.mode:
# grid`` with width=2000/height=2000 and no overlap then produces exactly these six
# tiles, so the pipeline performs one whole-image upload while the detector sees six
# tiles.  No real production image is checked in (see the acceptance items in Todo.md),
# so the tool validates the *pipeline* path and the CPU/CUDA agreement at production-like
# sizes.
ROI_ORIGINS = _roi_origins(ROI, columns=2, rows=3)
PRODUCTION_ROI_ORIGINS = [
    (PRODUCTION_BASE[1], PRODUCTION_BASE[0] + col * (PRODUCTION_ROI[1] + PRODUCTION_GAP_X))
    for col in range(6)
]

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
    template_path: Path | None = None,
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

    if template_path is not None:
        marker_rng = np.random.default_rng(20260915)
        marker = marker_rng.integers(0, 256, (64, 64), dtype=np.uint8)
        marker = cv2.GaussianBlur(marker, (3, 3), 0)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        anchor_x, anchor_y = PRODUCTION_ANCHOR
        bgr[anchor_y : anchor_y + 64, anchor_x : anchor_x + 64] = marker_bgr
        template_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(template_path), marker_bgr):
            raise SystemExit(f"failed to write {template_path}")

    canvas_path.parent.mkdir(parents=True, exist_ok=True)
    write_params = [cv2.IMWRITE_PNG_COMPRESSION, 1] if canvas_path.suffix.lower() == ".png" else []
    if not cv2.imwrite(str(canvas_path), bgr, write_params):
        raise SystemExit(f"failed to write {canvas_path}")
    del bgr


def _recipe(
    base: dict,
    dll_path: str,
    use_gpu: bool,
    roi: tuple[int, int] = ROI,
    template_path: Path | None = None,
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
    if template_path is None:
        recipe["tile"] = {
            "mode": "grid",
            "width": roi[1],
            "height": roi[0],
            "overlap_x": 0,
            "overlap_y": 0,
        }
    else:
        recipe["tile"] = {
            "mode": "grid",
            "template_path": str(template_path.resolve()),
            "search_x": 0,
            "search_y": 0,
            "search_w": 512,
            "search_h": 512,
            "match_threshold": 0.999,
            "offset_x": PRODUCTION_BASE[0] - PRODUCTION_ANCHOR[0],
            "offset_y": PRODUCTION_BASE[1] - PRODUCTION_ANCHOR[1],
            "rows": 1,
            "cols": 6,
            "roi_w": roi[1],
            "roi_h": roi[0],
            "gap_x": PRODUCTION_GAP_X,
            "gap_y": 0,
        }
    recipe["output"] = {
        key: False
        for key in (
            "save_overlay", "save_csv", "save_matrix_csv", "save_json", "save_ng_tiles"
        )
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
        tile = tile_result.get("tile", {}) or {}
        tile_metadata = tile.get("metadata", {}) or {}
        # Tile placement comes from the anchor, which may run on the device; the defect bboxes are
        # tile-local, so without this a shifted anchor would still compare as identical.
        decision.append(
            (
                tile.get("tile_id"), tile.get("x"), tile.get("y"), tile.get("width"),
                tile.get("height"), tuple(tile_metadata.get("match_bbox") or ()),
            )
        )
        if isinstance(tile_metadata.get("score"), (int, float)):
            drifting.append(("anchor_score", float(tile_metadata["score"])))
        for detector_result in tile_result.get("detectors", []):
            decision.append(
                (
                    tile.get("tile_id"),
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


def _timing_summary(samples: list[float]) -> dict:
    ordered = sorted(float(value) for value in samples)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _stage_values(results: list[dict], detector: bool = False) -> dict[str, list[float]]:
    collected: dict[str, list[float]] = {}
    for result in results:
        performance = result.get("execution", {}).get("performance", {})
        if detector:
            rows = {
                f"{detector_id}.{stage}": seconds * 1000.0
                for detector_id, stages in performance.get("detector_stages_sec", {}).items()
                for stage, seconds in stages.items()
            }
        else:
            rows = {
                stage: seconds * 1000.0
                for stage, seconds in performance.get("stages_sec", {}).items()
            }
        for name, milliseconds in rows.items():
            collected.setdefault(name, []).append(float(milliseconds))
    return collected


def _run_repeated(
    cpu_path: Path,
    gpu_path: Path,
    image: Path,
    work: Path,
    warmup: int,
    repetitions: int,
) -> tuple[list[float], list[dict], list[float], list[dict]]:
    cpu_pipeline = AOIPipeline(cpu_path, work / "cpu_out")
    with GpuExecutionSession.from_recipe_path(gpu_path) as gpu_session:
        gpu_pipeline = AOIPipeline(gpu_path, work / "gpu_out", gpu_session=gpu_session)
        for _ in range(warmup):
            cpu_pipeline.run(image)
            gpu_pipeline.run(image)

        cpu_times: list[float] = []
        gpu_times: list[float] = []
        cpu_results: list[dict] = []
        gpu_results: list[dict] = []
        for index in range(repetitions):
            order = (("cpu", cpu_pipeline), ("gpu", gpu_pipeline))
            if index % 2:
                order = tuple(reversed(order))
            pair = {}
            for backend, pipeline in order:
                started = time.perf_counter()
                result = pipeline.run(image)
                pair[backend] = ((time.perf_counter() - started) * 1000.0, result)
            cpu_ms, cpu_result = pair["cpu"]
            gpu_ms, gpu_result = pair["gpu"]
            cpu_times.append(cpu_ms)
            gpu_times.append(gpu_ms)
            cpu_results.append(cpu_result)
            gpu_results.append(gpu_result)
    return cpu_times, cpu_results, gpu_times, gpu_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", default=str(ROOT / "gpu" / "visionflow_cuda.dll"))
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--keep", action="store_true", help="keep the generated canvas")
    parser.add_argument("--work", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=1, help="unmeasured CPU/GPU warm-up pairs")
    parser.add_argument("--repetitions", type=int, default=3, help="measured CPU/GPU pairs")
    parser.add_argument(
        "--profile",
        choices=("quick", "production"),
        default="quick",
        help=(
            "quick: six 2000x2000 ROIs on a 4000x6000 canvas; "
            "production: six 12000h x 2000w ROIs on a 16384x13000 canvas, "
            "which needs ~609 MiB for decoded BGR pixels"
        ),
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.repetitions <= 0:
        parser.error("--warmup must be >= 0 and --repetitions must be > 0")

    work = args.work or Path(tempfile.mkdtemp(prefix="visionflow_production_"))
    canvas = work / "production_canvas.bmp"
    template_path = work / "production_anchor.png"
    if args.profile == "production":
        canvas_size, roi, roi_origins = (
            PRODUCTION_CANVAS,
            PRODUCTION_ROI,
            PRODUCTION_ROI_ORIGINS,
        )
        selected_template = template_path
    else:
        canvas_size, roi, roi_origins = CANVAS, ROI, ROI_ORIGINS
        selected_template = None
    print(
        f"profile: {args.profile}  canvas {canvas_size[0]}x{canvas_size[1]} (w x h), "
        f"{len(roi_origins)} ROIs {roi[0]}h x {roi[1]}w"
    )
    if not canvas.is_file() or not args.keep:
        started = time.perf_counter()
        _render_canvas(
            canvas,
            canvas=canvas_size,
            roi=roi,
            roi_origins=roi_origins,
            template_path=selected_template,
        )
        print(
            f"wrote {canvas} ({canvas.stat().st_size / 1e6:.1f} MB) in "
            f"{time.perf_counter() - started:.1f} s"
        )

    base = RecipeManager().load(ROOT / "recipes" / "PRODUCT_A_AOI_01.yaml")
    cpu_recipe = _recipe(
        base, args.dll, use_gpu=False, roi=roi, template_path=selected_template
    )
    gpu_recipe = _recipe(
        base, args.dll, use_gpu=True, roi=roi, template_path=selected_template
    )
    cpu_path = work / "cpu.yaml"
    gpu_path = work / "gpu.yaml"
    cpu_path.write_text(yaml.safe_dump(cpu_recipe, allow_unicode=True, sort_keys=False), encoding="utf-8")
    gpu_path.write_text(yaml.safe_dump(gpu_recipe, allow_unicode=True, sort_keys=False), encoding="utf-8")

    cpu_times, cpu_results, gpu_times, gpu_results = _run_repeated(
        cpu_path,
        gpu_path,
        canvas,
        work,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )
    cpu_timing = _timing_summary(cpu_times)
    gpu_timing = _timing_summary(gpu_times)
    cpu_ms = cpu_timing["median_ms"]
    gpu_ms = gpu_timing["median_ms"]
    cpu_result = cpu_results[-1]
    gpu_result = gpu_results[-1]

    cpu_final = cpu_result.get("final_result")
    gpu_final = gpu_result.get("final_result")
    cpu_summary = cpu_result.get("summary", {})
    gpu_summary = gpu_result.get("summary", {})
    print()
    print(
        f"CPU  end-to-end median={cpu_ms:9.1f} ms p95={cpu_timing['p95_ms']:9.1f} ms "
        f"final={cpu_final}  {cpu_summary}"
    )
    print(
        f"CUDA end-to-end median={gpu_ms:9.1f} ms p95={gpu_timing['p95_ms']:9.1f} ms "
        f"final={gpu_final}  {gpu_summary}"
    )
    print(f"speedup {cpu_ms / gpu_ms:.2f}x")

    cpu_normalised = _normalised(cpu_result)
    gpu_normalised = _normalised(gpu_result)
    strictly_identical = cpu_normalised == gpu_normalised
    comparisons = [_compare(cpu, gpu) for cpu, gpu in zip(cpu_results, gpu_results)]
    comparison = comparisons[-1]
    every_decision_equal = all(item["decision_equal"] for item in comparisons)
    print(f"normalised results strictly identical: {strictly_identical}")
    print(
        "decision-bearing fields identical (PASS/NG, defect count, type, bbox, area, "
        f"confidence, remaining metadata): {every_decision_equal} "
        f"({sum(item['decision_equal'] for item in comparisons)}/{len(comparisons)} runs)"
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
    if not every_decision_equal:
        print()
        print("DECISION-BEARING DIFFERENCE - the CUDA run is not equivalent:")
        cpu_tiles = cpu_normalised.get("tiles", [])
        gpu_tiles = gpu_normalised.get("tiles", [])
        for index, (cpu_tile, gpu_tile) in enumerate(zip(cpu_tiles, gpu_tiles)):
            if cpu_tile != gpu_tile:
                print(f"  tile {index}: CPU={json.dumps(cpu_tile, default=str)[:400]}")
                print(f"  tile {index}: GPU={json.dumps(gpu_tile, default=str)[:400]}")
                break

    # Build table-ready median/P95 rows from every measured result, not from the final run only.
    stage_comparison = []
    for scope, cpu_values, gpu_values in (
        ("detector", _stage_values(cpu_results, detector=True), _stage_values(gpu_results, detector=True)),
        ("pipeline", _stage_values(cpu_results), _stage_values(gpu_results)),
    ):
        for name in sorted(set(cpu_values) | set(gpu_values)):
            if name not in cpu_values or name not in gpu_values:
                continue
            cpu_row = _timing_summary(cpu_values[name])
            gpu_row = _timing_summary(gpu_values[name])
            stage_comparison.append(
                {
                    "scope": scope,
                    "stage": name,
                    "cpu_median_ms": cpu_row["median_ms"],
                    "cpu_p95_ms": cpu_row["p95_ms"],
                    "gpu_median_ms": gpu_row["median_ms"],
                    "gpu_p95_ms": gpu_row["p95_ms"],
                    "speedup": (
                        cpu_row["median_ms"] / gpu_row["median_ms"]
                        if gpu_row["median_ms"] else None
                    ),
                }
            )
    print()
    print(
        f"{'scope/stage':52s} {'CPU med':>10s} {'CPU p95':>10s} "
        f"{'GPU med':>10s} {'GPU p95':>10s} {'speedup':>9s}"
    )
    for row in stage_comparison:
        label = f"{row['scope']}/{row['stage']}"
        speedup = f"{row['speedup']:.2f}x" if row["speedup"] is not None else "n/a"
        print(
            f"{label:52s} {row['cpu_median_ms']:10.2f} {row['cpu_p95_ms']:10.2f} "
            f"{row['gpu_median_ms']:10.2f} {row['gpu_p95_ms']:10.2f} "
            f"{speedup:>9s}"
        )

    payload = {
        "kind": "pipeline_production",
        "profile": args.profile,
        "canvas": {"width": canvas_size[0], "height": canvas_size[1]},
        "roi": {"height": roi[0], "width": roi[1], "origins": roi_origins},
        "cpu_ms": cpu_ms,
        "gpu_ms": gpu_ms,
        "speedup": cpu_ms / gpu_ms if gpu_ms else None,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "cpu_timing": cpu_timing,
        "gpu_timing": gpu_timing,
        "cpu_samples_ms": cpu_times,
        "gpu_samples_ms": gpu_times,
        "cpu_final": cpu_final,
        "gpu_final": gpu_final,
        "cpu_summary": cpu_summary,
        "gpu_summary": gpu_summary,
        "normalised_identical": strictly_identical,
        "decision_fields_identical": every_decision_equal,
        "decision_equal_runs": sum(item["decision_equal"] for item in comparisons),
        "drifting_diagnostics": comparison["drift_counts"],
        "worst_diagnostic_drift": comparison["worst_drift"],
        "device_host_split": split,
        "resident_image": resident,
        "tiling": tiling,
        "stage_comparison": stage_comparison,
        "gpu_metrics": gpu_result.get("execution", {}).get("gpu", {}).get("metrics", {}),
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0 if every_decision_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
