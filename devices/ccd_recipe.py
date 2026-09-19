from __future__ import annotations

import math
from typing import Any

from devices.ccd_models import (
    EXPOSURE_RANGE,
    GAIN_RANGE,
    LENGTH_LINES_RANGE,
    LINE_RATE_HZ_RANGE,
    AcquisitionSettings,
    CameraRecipeSettings,
    TriggerMode,
    TriggerSettings,
)

# ============================================================
# Recipe `camera` section codec (product-level CCD settings).
#
# camera:
#   exposure_time: 1200.0
#   gain: 1.0
#   length_lines: 720
#   internal_line_rate_hz: 30
#   trigger:
#     mode: continuous | external_trigger | software_trigger
#     external_frame_one_frame: false
#     compare_follows_encoder: false
#     set_encoder_on_trigger: false
#   auto_save:
#     external_one_frame: false
#     software_trigger: false
#
# The section is optional; a Recipe without it never changes the camera settings.
# ============================================================

CAMERA_SECTION = "camera"
_ACQUISITION_KEYS = ("exposure_time", "gain", "length_lines", "internal_line_rate_hz")
_TRIGGER_FLAGS = ("external_frame_one_frame", "compare_follows_encoder", "set_encoder_on_trigger")
_AUTO_SAVE_KEYS = ("external_one_frame", "software_trigger")


def _mapping(value: Any, path: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"Recipe {path} must be a mapping.")
    return value


def _reject_unknown(section: dict, allowed: tuple[str, ...], path: str) -> None:
    unknown = set(section) - set(allowed)
    if unknown:
        raise ValueError(f"Recipe {path} has unknown keys: {', '.join(sorted(map(str, unknown)))}")


def _number(section: dict, key: str, bounds: tuple, path: str, integer: bool):
    if key not in section:
        raise ValueError(f"Recipe {path}.{key} is required.")
    value = section[key]
    kind = "an integer" if integer else "a number"
    valid_type = isinstance(value, int) if integer else isinstance(value, (int, float))
    if isinstance(value, bool) or not valid_type or not math.isfinite(float(value)):
        raise ValueError(f"Recipe {path}.{key} must be {kind}.")
    low, high = bounds
    if not low <= value <= high:
        raise ValueError(f"Recipe {path}.{key} must be between {low} and {high}.")
    return int(value) if integer else float(value)


def _flag(section: dict, key: str, path: str) -> bool:
    value = section.get(key, False)
    if not isinstance(value, bool):
        raise ValueError(f"Recipe {path}.{key} must be true or false.")
    return value


def parse_camera_section(section: Any) -> CameraRecipeSettings:
    section = _mapping(section, CAMERA_SECTION)
    _reject_unknown(section, (*_ACQUISITION_KEYS, "trigger", "auto_save"), CAMERA_SECTION)
    acquisition = AcquisitionSettings(
        exposure_time=_number(section, "exposure_time", EXPOSURE_RANGE, CAMERA_SECTION, integer=False),
        gain=_number(section, "gain", GAIN_RANGE, CAMERA_SECTION, integer=False),
        length_lines=_number(section, "length_lines", LENGTH_LINES_RANGE, CAMERA_SECTION, integer=True),
        internal_line_rate_hz=_number(section, "internal_line_rate_hz", LINE_RATE_HZ_RANGE, CAMERA_SECTION, integer=True),
    )

    trigger_path = f"{CAMERA_SECTION}.trigger"
    trigger_section = _mapping(section.get("trigger"), trigger_path)
    _reject_unknown(trigger_section, ("mode", *_TRIGGER_FLAGS), trigger_path)
    try:
        mode = TriggerMode(trigger_section.get("mode"))
    except ValueError:
        modes = ", ".join(item.value for item in TriggerMode)
        raise ValueError(f"Recipe {trigger_path}.mode must be one of: {modes}.") from None
    trigger = TriggerSettings(mode, *(_flag(trigger_section, key, trigger_path) for key in _TRIGGER_FLAGS))
    normalized = trigger.normalized()
    if normalized != trigger:
        conflicts = [key for key in _TRIGGER_FLAGS if getattr(trigger, key) != getattr(normalized, key)]
        raise ValueError(
            f"Recipe {trigger_path} options are not allowed with mode {mode.value}: {', '.join(conflicts)}."
        )

    auto_save_path = f"{CAMERA_SECTION}.auto_save"
    auto_save = _mapping(section.get("auto_save", {}), auto_save_path)
    _reject_unknown(auto_save, _AUTO_SAVE_KEYS, auto_save_path)
    return CameraRecipeSettings(
        acquisition=acquisition,
        trigger=trigger,
        auto_save_external_one_frame=_flag(auto_save, "external_one_frame", auto_save_path),
        auto_save_software_trigger=_flag(auto_save, "software_trigger", auto_save_path),
    )


def camera_settings_from_recipe(recipe: dict | None) -> CameraRecipeSettings | None:
    section = (recipe or {}).get(CAMERA_SECTION)
    return None if section is None else parse_camera_section(section)


def camera_section(settings: CameraRecipeSettings) -> dict:
    settings = settings.normalized()
    acquisition = settings.acquisition
    trigger = settings.trigger
    return {
        "exposure_time": float(acquisition.exposure_time),
        "gain": float(acquisition.gain),
        "length_lines": int(acquisition.length_lines),
        "internal_line_rate_hz": int(acquisition.internal_line_rate_hz),
        "trigger": {
            "mode": trigger.mode.value,
            "external_frame_one_frame": trigger.external_frame_one_frame,
            "compare_follows_encoder": trigger.compare_follows_encoder,
            "set_encoder_on_trigger": trigger.set_encoder_on_trigger,
        },
        "auto_save": {
            "external_one_frame": settings.auto_save_external_one_frame,
            "software_trigger": settings.auto_save_software_trigger,
        },
    }
