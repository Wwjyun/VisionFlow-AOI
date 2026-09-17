"""CPU versus GPU comparison of one image and one Recipe, shared by the GUI and the production benchmark."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Callable

from core.logging_system import LogMixin

# The device float32 Gaussian is mathematically equivalent, not bit-identical, so four
# residual-derived diagnostics drift in their last bits.  They are reported separately
# instead of being hidden, and the decision-bearing fields are compared strictly.
DRIFTING_METADATA = {
    "mad",
    "residual_median",
    "residual_threshold",
    "robust_noise_sigma",
}

# Fields that exist precisely to say *which* backend produced the record.  They must
# differ between the CPU and CUDA runs, so comparing them as decision-bearing would
# report a difference on every defect and hide a real mismatch.
BACKEND_PROVENANCE = {
    "background_backend",
    "residual_backend",
    "background_precision_note",
    "component_backend",
}

COMPARISON_OUTPUT_OVERRIDES = {
    "save_overlay": False,
    "save_ng_tiles": False,
    "save_csv": False,
    "save_matrix_csv": False,
    "save_json": False,
    "save_debug_images": False,
}

STAGE_ORDER = ("image_load", "initialization", "tiling", "detectors_total", "aggregation", "reporting_total")


def normalised_result(result: dict) -> dict:
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


def split_decision_fields(result: dict):
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
                    if name in DRIFTING_METADATA and isinstance(value, (int, float)):
                        drifting.append((name, float(value)))
                    elif name in BACKEND_PROVENANCE:
                        continue
                    else:
                        decision.append((name, value))
    return decision, drifting, worst


def compare_decisions(cpu_result: dict, gpu_result: dict) -> dict:
    cpu_decision, cpu_drift, _ = split_decision_fields(cpu_result)
    gpu_decision, gpu_drift, _ = split_decision_fields(gpu_result)
    decision_equal = cpu_decision == gpu_decision and cpu_result.get("final_result") == gpu_result.get("final_result")
    drift_counts = {}
    worst_drift = 0.0
    if len(cpu_drift) != len(gpu_drift):
        decision_equal = False
    for (name, cpu_value), (gpu_name, gpu_value) in zip(cpu_drift, gpu_drift):
        if name != gpu_name:
            decision_equal = False
            break
        delta = abs(cpu_value - gpu_value)
        worst_drift = max(worst_drift, delta)
        if delta > 0.0:
            drift_counts[name] = drift_counts.get(name, 0) + 1
    first_difference = None
    if cpu_decision != gpu_decision:
        for index, (cpu_field, gpu_field) in enumerate(zip(cpu_decision, gpu_decision)):
            if cpu_field != gpu_field:
                first_difference = {"index": index, "cpu": repr(cpu_field), "gpu": repr(gpu_field)}
                break
        else:
            first_difference = {"index": min(len(cpu_decision), len(gpu_decision)),
                                "cpu": f"{len(cpu_decision)} fields", "gpu": f"{len(gpu_decision)} fields"}
    return {
        "decision_equal": decision_equal,
        "drift_counts": drift_counts,
        "worst_drift": worst_drift,
        "drifting_field_count": len(cpu_drift),
        "first_difference": first_difference,
    }


def stage_comparison(cpu_result: dict, gpu_result: dict) -> list[dict]:
    """End-to-end plus pipeline and Detector stages with CPU/GPU milliseconds and speedup."""
    rows = []

    def add(scope: str, name: str, cpu_seconds, gpu_seconds) -> None:
        cpu_ms = float(cpu_seconds or 0.0) * 1000.0
        gpu_ms = float(gpu_seconds or 0.0) * 1000.0
        rows.append({
            "scope": scope,
            "stage": name,
            "cpu_ms": round(cpu_ms, 3),
            "gpu_ms": round(gpu_ms, 3),
            "speedup": round(cpu_ms / gpu_ms, 2) if gpu_ms > 0 else None,
        })

    cpu_perf = (cpu_result.get("execution", {}) or {}).get("performance", {}) or {}
    gpu_perf = (gpu_result.get("execution", {}) or {}).get("performance", {}) or {}
    add("total", "end_to_end", cpu_perf.get("end_to_end_sec"), gpu_perf.get("end_to_end_sec"))
    cpu_stages = cpu_perf.get("stages_sec", {}) or {}
    gpu_stages = gpu_perf.get("stages_sec", {}) or {}
    for stage in STAGE_ORDER:
        if stage in cpu_stages or stage in gpu_stages:
            add("pipeline", stage, cpu_stages.get(stage), gpu_stages.get(stage))
    cpu_detectors = cpu_perf.get("detectors_sec", {}) or {}
    gpu_detectors = gpu_perf.get("detectors_sec", {}) or {}
    for detector_id in sorted(set(cpu_detectors) | set(gpu_detectors)):
        add("detector", detector_id, cpu_detectors.get(detector_id), gpu_detectors.get(detector_id))
    return rows


def actual_gpu_backend(result: dict) -> tuple[bool, str]:
    """Whether any Detector or tiling step actually ran on CUDA, plus the first fallback reason."""
    gpu = (result.get("execution", {}) or {}).get("gpu", {}) or {}
    statuses = [gpu.get("tiling", {}) or {}] + [status or {} for status in (gpu.get("detectors", {}) or {}).values()]
    active = any(status.get("active") for status in statuses)
    reason = next((str(status.get("fallback_reason")) for status in statuses if status.get("fallback_reason")), "")
    return active, reason


class BackendComparison(LogMixin):
    """Run one image through the Recipe as requested and forced to CPU, then compare decisions.

    The Recipe file is never modified: the CPU run uses the pipeline's ``gpu_mode_override`` and
    both runs write no report files. When the requested run did not actually use CUDA the report
    says so instead of presenting a CPU-versus-CPU comparison as a GPU result.
    """

    def __init__(self, pipeline_factory=None):
        if pipeline_factory is None:
            from core.pipeline import AOIPipeline

            pipeline_factory = AOIPipeline
        self._pipeline_factory = pipeline_factory

    def run(
        self,
        recipe_path: Path,
        image_path: Path,
        output_dir: Path,
        gpu_session=None,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> dict:
        def report(percent: int, message: str) -> None:
            if progress_callback is not None:
                progress_callback(int(percent), message)

        report(5, "GPU 設定執行中（不輸出檔案）")
        started = time.perf_counter()
        gpu_result = self._pipeline_factory(
            Path(recipe_path), Path(output_dir), output_overrides=dict(COMPARISON_OUTPUT_OVERRIDES),
            gpu_session=gpu_session,
        ).run(Path(image_path))
        gpu_wall = time.perf_counter() - started
        report(50, "CPU 參考執行中（不輸出檔案）")
        started = time.perf_counter()
        cpu_result = self._pipeline_factory(
            Path(recipe_path), Path(output_dir), output_overrides=dict(COMPARISON_OUTPUT_OVERRIDES),
            gpu_mode_override="cpu",
        ).run(Path(image_path))
        cpu_wall = time.perf_counter() - started
        report(95, "比對判定欄位")
        gpu_active, fallback_reason = actual_gpu_backend(gpu_result)
        comparison = compare_decisions(cpu_result, gpu_result)
        summary = {
            "image_name": Path(image_path).name,
            "recipe_name": gpu_result.get("recipe_name", ""),
            "gpu_active": gpu_active,
            "gpu_fallback_reason": fallback_reason,
            "cpu_final": cpu_result.get("final_result"),
            "gpu_final": gpu_result.get("final_result"),
            "cpu_defects": int((cpu_result.get("summary", {}) or {}).get("defect_count", 0)),
            "gpu_defects": int((gpu_result.get("summary", {}) or {}).get("defect_count", 0)),
            "cpu_wall_ms": round(cpu_wall * 1000.0, 1),
            "gpu_wall_ms": round(gpu_wall * 1000.0, 1),
            "stages": stage_comparison(cpu_result, gpu_result),
            **comparison,
        }
        self.logger.info("CPU/GPU comparison: %s", {k: v for k, v in summary.items() if k != "stages"})
        report(100, "CPU／GPU 對照完成")
        return summary
