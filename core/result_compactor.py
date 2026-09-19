from __future__ import annotations


TOP_LEVEL_KEYS = (
    "image_name",
    "recipe_name",
    "machine_id",
    "product_id",
    "recipe_version",
    "final_result",
    "summary",
    "outputs",
    "duration_sec",
    "execution",
    "source",
)

TILE_KEYS = ("tile_id", "x", "y", "width", "height", "row", "col")
DETECTOR_KEYS = ("detector_id", "pass", "score", "execution")
DEFECT_KEYS = ("type", "bbox_global", "bbox_local", "area", "confidence")


def compact_inspection_result(result: dict) -> dict:
    """Keep only GUI batch/monitor fields; full detail remains in report JSON."""
    compact = {key: result.get(key) for key in TOP_LEVEL_KEYS if key in result}
    compact["summary"] = dict(result.get("summary", {}) or {})
    compact["outputs"] = dict(result.get("outputs", {}) or {})
    compact["tiles"] = [_compact_tile_result(tile_result) for tile_result in result.get("tiles", []) or []]
    return compact


def _compact_tile_result(tile_result: dict) -> dict:
    tile = tile_result.get("tile", {}) or {}
    compact_tile = {key: tile.get(key) for key in TILE_KEYS if key in tile}
    compact_result = {
        "tile": compact_tile,
        "result": tile_result.get("result", "PASS"),
        "detectors": [],
    }
    for detector_result in tile_result.get("detectors", []) or []:
        compact_result["detectors"].append(_compact_detector_result(detector_result))
    return compact_result


def _compact_detector_result(detector_result: dict) -> dict:
    compact = {key: detector_result.get(key) for key in DETECTOR_KEYS if key in detector_result}
    compact["defects"] = [
        _compact_defect(defect)
        for defect in detector_result.get("defects", []) or []
    ]
    return compact


def _compact_defect(defect: dict) -> dict:
    compact = {key: defect.get(key) for key in DEFECT_KEYS if key in defect}
    metadata = defect.get("metadata", {}) or {}
    reason = metadata.get("reason")
    if reason:
        compact["metadata"] = {"reason": reason}
    return compact
