"""One-time import of the reference C# application's ``settings.ini`` (Todo P11).

The authoritative schema is ``xx_ccd/CameraCaptureApp/Services/SettingsService.cs``: a flat,
case-insensitive ``Key=Value`` dictionary where blank, ``;``, ``#`` and ``[section]`` lines are
ignored, the first ``=`` splits the line, both sides are trimmed, and values are parsed with
``CultureInfo.InvariantCulture``. Every key that is absent, unknown or unparsable keeps the C#
default: a bad value is reported as a Traditional-Chinese warning instead of raising.

Machine-level keys become :class:`~devices.ccd_models.CcdMachineSettings` (the machine settings
store); product-level keys become :class:`~devices.ccd_models.CameraRecipeSettings` (the Recipe
``camera`` section). The two levels never cross.

Deliberate deviations from the C# reference, all reported to the operator:

* ``MeterWheelMultipleRate`` outside ``0``/``1``/``2`` warns instead of being silently reset to X4.
* ``TriggerMode=SingleFrame`` is not ported (Todo P11 "已確認決策"), so the mode stays ``Continuous``.
* An absent key keeps the VisionFlow value-object default; only a key that is present with an
  unparsable value keeps the C# parse fallback. ``ImageSaveFormat`` is the single imported key where
  the two differ (C# ``Png`` vs VisionFlow ``Bmp``, the production handoff format), so an ini that
  does not mention it imports as ``BMP`` and an unparsable one imports as ``PNG`` with a warning
  that names both.
* ``InternalLineRate`` is a decimal in C# but ``internal_line_rate_hz`` is an integer, so a
  fractional value is truncated with a warning.
* ``SaveSettings.max_concurrent_saves`` has no ini counterpart (the C# app used a fixed 5), and the
  keys of the truncated C# feature set (``Width``/``Height``, rolling capture, grayscale waveform,
  ...) are reported per key rather than dropped silently.

Reading is UTF-8 with ``utf-8-sig`` semantics (the C# writer emits UTF-8 with a BOM) and every
failure raises :class:`CcdSettingsImportError`; nothing here touches Qt or hardware.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from devices.ccd_models import (
    EXTENSION_CHANNEL_COUNT,
    AcquisitionSettings,
    CameraConnectionSettings,
    CameraRecipeSettings,
    CcdMachineSettings,
    ExtensionCompareChannel,
    ImageSaveFormat,
    MeterWheelSettings,
    MultipleRate,
    SaveSettings,
    TriggerMode,
    TriggerSettings,
)

IMPORT_SCHEMA = "visionflow-ccd-settings-import/v1"
SETTINGS_FILE_NAME = "settings.ini"
SETTINGS_PATH_ENV = "VISIONFLOW_CCD_SETTINGS_INI"

# Keys the C# app persists that VisionFlow deliberately does not import. Each present key produces
# exactly one warning naming the key and the reason, so nothing is dropped silently.
UNPORTED_KEYS = (
    ("CameraName", "相機名稱由 Sapera 連線時的 server／resource 決定，VisionFlow 不另存名稱。"),
    ("ServerIndex", "Sapera 位置以 ServerName 與 ResourceIndex 表示，不使用 ServerIndex。"),
    ("DeviceFeatureConfigFilePath", "DeviceFeature 不使用 CCF 檔，只有 ConfigFilePath 會匯入。"),
    ("Width", "影像寬度由相機 buffer 決定，不存入設定。"),
    ("Height", "影像高度由相機 buffer 決定，不存入設定。"),
    ("RollingCaptureEnabled", "滾動式拍照已列為暫不移植。"),
    ("RollingCaptureFrameCount", "滾動式拍照已列為暫不移植。"),
    ("RollingCaptureDirection", "滾動式拍照已列為暫不移植。"),
    ("FrameRate", "線掃相機的取像速率由 InternalLineRate 決定，FrameRate 只用於面掃相機。"),
    ("PixelFormat", "像素格式固定為 Mono8，由相機 buffer 決定。"),
    ("AutoConnect", "VisionFlow 不自動連線相機，連線由 CCD 頁面操作。"),
    ("AutoSave", "自動存圖改由 AutoSaveOnExternalTriggerOneFrame 與 AutoSaveOnSoftwareTriggerFrame 決定。"),
    ("FileNamePattern", "檔名由 VisionFlow 的存圖規則決定，不沿用 C# 的樣式。"),
)
_UNPORTED_REASONS = dict(UNPORTED_KEYS)

_INT32_MIN = -2_147_483_648
_INT32_MAX = 2_147_483_647
# System.Decimal.MaxValue; decimal.TryParse fails above it, so the C# default is kept instead.
_DECIMAL_MAX = 79_228_162_514_264_337_593_543_950_335

_INT_RE = re.compile(r"^[+-]?\d+$")
# NumberStyles.Number: optional sign, optional thousands separators, optional decimal point.
_DECIMAL_RE = re.compile(r"^[+-]?(?:(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d*)?|\.\d+)$")
_LINE_RE = re.compile(r"\r\n|\r|\n")
_HIGH_BIT_MASK = ~0xFF

_SINGLE_FRAME = object()

_TRIGGER_MODES: dict[str, Any] = {
    # C# `Enum.TryParse(value, true, ...)` matches member names case-insensitively.
    "continuous": TriggerMode.CONTINUOUS,
    "externaltrigger": TriggerMode.EXTERNAL,
    "softwaretrigger": TriggerMode.SOFTWARE,
    "singleframe": _SINGLE_FRAME,
    # VisionFlow value aliases, accepted for hand-written files.
    "external_trigger": TriggerMode.EXTERNAL,
    "software_trigger": TriggerMode.SOFTWARE,
}

_IMAGE_SAVE_FORMATS = {
    "png": ImageSaveFormat.PNG,
    "tif": ImageSaveFormat.TIF,
    "uncompressedtif": ImageSaveFormat.TIF_UNCOMPRESSED,
    "tif_uncompressed": ImageSaveFormat.TIF_UNCOMPRESSED,
}

_MULTIPLE_RATES = {0: MultipleRate.X4, 1: MultipleRate.X2, 2: MultipleRate.X1}

# `CameraSettings.CreateDefault()` uses `ImageSaveFormat.Png`; `SaveSettings` defaults to `Bmp`.
_C_SHARP_IMAGE_SAVE_FORMAT = ImageSaveFormat.PNG

_VALUE_LABELS = {
    MultipleRate.X4: "X4",
    MultipleRate.X2: "X2",
    MultipleRate.X1: "X1",
    TriggerMode.CONTINUOUS: "Continuous",
    TriggerMode.EXTERNAL: "ExternalTrigger",
    TriggerMode.SOFTWARE: "SoftwareTrigger",
    ImageSaveFormat.BMP: "Bmp",
    ImageSaveFormat.PNG: "Png",
    ImageSaveFormat.TIF: "Tif",
    ImageSaveFormat.TIF_UNCOMPRESSED: "UncompressedTif",
}


class CcdSettingsImportError(ValueError):
    """`settings.ini` could not be read; the message is Traditional Chinese and names the path."""


@dataclass(frozen=True)
class ImportedCcdSettings:
    """One imported `settings.ini`, split into the machine-level and product-level halves."""

    machine: CcdMachineSettings
    product: CameraRecipeSettings
    warnings: tuple[str, ...]
    source: str


def _display(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return _VALUE_LABELS.get(value, str(value))


def _key(name: str) -> str:
    """Case-insensitive key, as in the C# `StringComparer.OrdinalIgnoreCase` dictionary."""
    return name.casefold()


def _parse_int32(text: str) -> int | None:
    """Mirror `int.TryParse(value, NumberStyles.Integer, InvariantCulture)` including Int32 overflow."""
    if not _INT_RE.match(text):
        return None
    value = int(text)
    return value if _INT32_MIN <= value <= _INT32_MAX else None


def _parse_decimal(text: str) -> float | None:
    """Mirror `decimal.TryParse(value, NumberStyles.Number, InvariantCulture)`."""
    if not _DECIMAL_RE.match(text):
        return None
    value = float(text.replace(",", ""))
    return value if -_DECIMAL_MAX <= value <= _DECIMAL_MAX else None


def _parse_bool(text: str) -> bool | None:
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def read_ini_values(text: str) -> dict[str, str]:
    """Mirror `SettingsService.ReadIniValues`; the returned keys are casefolded."""
    values: dict[str, str] = {}
    for raw_line in _LINE_RE.split(text):
        line = raw_line.strip()
        if not line or line.startswith((";", "#", "[")):
            continue
        separator = line.find("=")
        if separator <= 0:
            continue
        values[_key(line[:separator].strip())] = line[separator + 1 :].strip()
    return values


def parse_ccd_settings_ini(text: str, *, source: str = "<memory>") -> ImportedCcdSettings:
    """Convert `settings.ini` text into machine-level and product-level CCD settings.

    Keys are read in the order `SettingsService.Load` reads them, so the warnings stay deterministic.
    A bad value never raises: it keeps the C# default and appends one warning.
    """
    if text[:1] == "\ufeff":
        text = text[1:]
    values = read_ini_values(text)
    warnings: list[str] = []
    parsed: dict[str, Any] = {}
    seen_imported = False

    def note(message: str) -> None:
        warnings.append(message)

    def present(key: str) -> bool:
        return _key(key) in values

    def unparsable(key: str, kind: str, fallback: Any) -> None:
        note(f"設定值 {key}={values[_key(key)]!r} 無法解析為{kind}，已沿用預設值 {_display(fallback)}。")

    def unported(key: str) -> None:
        nonlocal seen_imported
        if present(key):
            seen_imported = True
            note(f"{key} 未匯入：{_UNPORTED_REASONS[key]}")

    def get_string(key: str, default: str) -> str:
        nonlocal seen_imported
        if not present(key):
            return default
        seen_imported = True
        value = values[_key(key)]
        parsed[key] = value
        return value

    def get_int(key: str, default: int) -> int:
        nonlocal seen_imported
        if not present(key):
            return default
        seen_imported = True
        value = _parse_int32(values[_key(key)])
        if value is None:
            unparsable(key, "整數", default)
            return default
        parsed[key] = value
        return value

    def get_decimal(key: str, default: float) -> float:
        nonlocal seen_imported
        if not present(key):
            return default
        seen_imported = True
        value = _parse_decimal(values[_key(key)])
        if value is None:
            unparsable(key, "數字", default)
            return default
        parsed[key] = value
        return value

    def get_bool(key: str, default: bool) -> bool:
        nonlocal seen_imported
        if not present(key):
            return default
        seen_imported = True
        value = _parse_bool(values[_key(key)])
        if value is None:
            unparsable(key, "true／false", default)
            return default
        parsed[key] = value
        return value

    def get_trigger_mode(key: str, default: TriggerMode) -> TriggerMode:
        nonlocal seen_imported
        if not present(key):
            return default
        seen_imported = True
        raw = values[_key(key)]
        value = _TRIGGER_MODES.get(raw.casefold())
        if value is None:
            unparsable(key, "觸發模式", default)
            return default
        if value is _SINGLE_FRAME:
            note(f"設定值 {key}=SingleFrame 暫不移植（Todo P11 已確認決策），已沿用 TriggerMode=Continuous。")
            return default
        parsed[key] = value
        return value

    def get_image_format(key: str) -> ImageSaveFormat:
        """An absent key keeps the VisionFlow default; an unparsable one keeps the C# default."""
        nonlocal seen_imported
        if not present(key):
            return SaveSettings().image_format
        seen_imported = True
        raw = values[_key(key)]
        value = _IMAGE_SAVE_FORMATS.get(raw.casefold())
        if value is None:
            note(
                f"設定值 {key}={raw!r} 無法解析為存圖格式，"
                f"已沿用 C# 預設值 {_display(_C_SHARP_IMAGE_SAVE_FORMAT)}（VisionFlow 預設為 Bmp）。"
            )
            return _C_SHARP_IMAGE_SAVE_FORMAT
        parsed[key] = value
        return value

    def get_bitmask(key: str) -> int:
        """Return the raw bitmask, or 0 when the key is absent or unparsable."""
        nonlocal seen_imported
        if not present(key):
            return 0
        seen_imported = True
        raw = values[_key(key)]
        value = _parse_int32(raw)
        if value is None:
            unparsable(key, "整數位元遮罩", 0)
            return 0
        parsed[key] = value
        return value

    def get_channel_values(key: str, minimum: int, maximum: int) -> dict[int, int]:
        """Parse the comma-separated CMP0–7 list; every problem becomes one warning for the key."""
        nonlocal seen_imported
        if not present(key):
            return {}
        seen_imported = True
        raw = values[_key(key)]
        parts = [part.strip() for part in raw.split(",")] if raw.strip() else []
        issues: list[str] = []
        result: dict[int, int] = {}
        for index in range(min(len(parts), EXTENSION_CHANNEL_COUNT)):
            number = _parse_int32(parts[index])
            if number is None:
                issues.append(f"索引 {index}（{parts[index]!r}）無法解析，該通道沿用預設值 0")
                continue
            clamped = max(minimum, min(maximum, number))
            if clamped != number:
                issues.append(f"索引 {index} 的 {number} 超出範圍 {minimum}–{maximum}，已調整為 {clamped}")
            result[index] = clamped
        if len(parts) < EXTENSION_CHANNEL_COUNT:
            issues.append(
                f"只提供 {len(parts)} 個值（需要 {EXTENSION_CHANNEL_COUNT} 個），"
                f"通道 {len(parts)}–{EXTENSION_CHANNEL_COUNT - 1} 沿用預設值 0"
            )
        elif len(parts) > EXTENSION_CHANNEL_COUNT:
            issues.append(f"提供 {len(parts)} 個值，只使用前 {EXTENSION_CHANNEL_COUNT} 個")
        if issues:
            note(f"設定值 {key}={raw!r} 有 {len(issues)} 項問題：" + "；".join(issues) + "。")
        return result

    # ---- read every key in the order `SettingsService.Load` reads it -------------------------
    unported("CameraName")
    config_file_path = get_string("ConfigFilePath", "")
    server_name = get_string("ServerName", "")
    unported("ServerIndex")
    resource_index = get_int("ResourceIndex", 0)
    device_feature_server_name = get_string("DeviceFeatureServerName", "")
    unported("DeviceFeatureConfigFilePath")
    device_feature_resource_index = get_int("DeviceFeatureResourceIndex", -1)
    unported("Width")
    unported("Height")
    length_lines = get_int("Length", 720)
    unported("RollingCaptureEnabled")
    unported("RollingCaptureFrameCount")
    unported("RollingCaptureDirection")
    exposure_time = get_decimal("ExposureTime", 1200.0)
    gain = get_decimal("Gain", 1.0)
    internal_line_rate = get_decimal("InternalLineRate", 30.0)
    unported("FrameRate")
    unported("PixelFormat")
    trigger_mode = get_trigger_mode("TriggerMode", TriggerMode.CONTINUOUS)
    external_one_frame = get_bool("ExternalFrameTriggerOneFrame", False)
    compare_from_encoder = get_bool("ExternalFrameTriggerOneFrameCompareFromEncoder", False)
    set_encoder_on_trigger = get_bool("ExternalFrameTriggerOneFrameSetEncoderOnTrigger", False)
    unported("AutoConnect")
    unported("AutoSave")
    auto_save_external = get_bool("AutoSaveOnExternalTriggerOneFrame", False)
    auto_save_software = get_bool("AutoSaveOnSoftwareTriggerFrame", False)
    save_folder = get_string("SaveFolder", "")
    unported("FileNamePattern")
    image_format = get_image_format("ImageSaveFormat")
    meter_compare_increment = get_int("MeterWheelCompareIncrement", 0)
    meter_encoder_value = get_int("MeterWheelEncoderValue", 0)
    meter_compare_value = get_int("MeterWheelCompareValue", 0)
    meter_card_id = get_int("MeterWheelCardId", 0)
    meter_multiple_rate = get_int("MeterWheelMultipleRate", 0)
    multiple_rate = _MULTIPLE_RATES.get(meter_multiple_rate, MultipleRate.X4)
    if "MeterWheelMultipleRate" in parsed and multiple_rate is MultipleRate.X4 and meter_multiple_rate != 0:
        note(
            f"設定值 MeterWheelMultipleRate={meter_multiple_rate} 不是有效倍頻"
            "（0=X4、1=X2、2=X1），已沿用預設 X4。"
        )
    meter_reverse_direction = get_bool("MeterWheelReverseDirection", False)
    meter_cmp_out_width = get_int("MeterWheelCmpOutWidth", 0)
    meter_mask = get_bitmask("MeterWheelExtensionCompareMask")
    meter_offsets = get_channel_values("MeterWheelExtensionCompareOffsets", -32_768, 32_767)
    meter_pulse_widths = get_channel_values("MeterWheelExtensionComparePulseWidths", 0, 65_535)
    meter_output_states = get_bitmask("MeterWheelExtensionCompareOutputStates")

    for key, raw_mask in (
        ("MeterWheelExtensionCompareMask", meter_mask),
        ("MeterWheelExtensionCompareOutputStates", meter_output_states),
    ):
        if raw_mask & _HIGH_BIT_MASK:
            note(f"設定值 {key}={raw_mask} 含 CMP0–7 以外的位元，已忽略。")

    # ---- machine-level value objects --------------------------------------------------------
    channels = tuple(
        ExtensionCompareChannel(
            masked=bool(meter_mask >> index & 1),
            offset=meter_offsets.get(index, 0),
            pulse_width=meter_pulse_widths.get(index, 0),
            output_state=bool(meter_output_states >> index & 1),
        )
        for index in range(EXTENSION_CHANNEL_COUNT)
    )
    machine = CcdMachineSettings(
        connection=CameraConnectionSettings(
            server_name=server_name,
            resource_index=resource_index,
            config_file_path=config_file_path,
            device_feature_server_name=device_feature_server_name,
            device_feature_resource_index=device_feature_resource_index,
        ),
        meter_wheel=MeterWheelSettings(
            card_id=meter_card_id,
            compare_increment=meter_compare_increment,
            multiple_rate=multiple_rate,
            reverse_direction=meter_reverse_direction,
            cmp_out_width=meter_cmp_out_width,
            encoder_value=meter_encoder_value,
            compare_value=meter_compare_value,
            extension_channels=channels,
        ),
        save=SaveSettings(folder=save_folder, image_format=image_format),
    ).normalized()

    # ---- product-level value objects (Recipe `camera` section) ------------------------------
    line_rate_hz = int(internal_line_rate)
    if line_rate_hz != internal_line_rate:
        note(
            f"設定值 InternalLineRate={_display(internal_line_rate)} 有小數，"
            f"線速率為整數，已取整為 {line_rate_hz}。"
        )
    product = CameraRecipeSettings(
        acquisition=AcquisitionSettings(exposure_time, gain, length_lines, line_rate_hz),
        trigger=TriggerSettings(trigger_mode, external_one_frame, compare_from_encoder, set_encoder_on_trigger),
        auto_save_external_one_frame=auto_save_external,
        auto_save_software_trigger=auto_save_software,
    ).normalized()

    # ---- values the value objects clamped after parsing -------------------------------------
    for key, before, after in (
        ("ResourceIndex", parsed.get("ResourceIndex"), machine.connection.resource_index),
        (
            "DeviceFeatureResourceIndex",
            parsed.get("DeviceFeatureResourceIndex"),
            machine.connection.device_feature_resource_index,
        ),
        ("MeterWheelCardId", parsed.get("MeterWheelCardId"), machine.meter_wheel.card_id),
        ("MeterWheelCompareIncrement", parsed.get("MeterWheelCompareIncrement"), machine.meter_wheel.compare_increment),
        ("MeterWheelEncoderValue", parsed.get("MeterWheelEncoderValue"), machine.meter_wheel.encoder_value),
        ("MeterWheelCompareValue", parsed.get("MeterWheelCompareValue"), machine.meter_wheel.compare_value),
        ("MeterWheelCmpOutWidth", parsed.get("MeterWheelCmpOutWidth"), machine.meter_wheel.cmp_out_width),
        ("Length", parsed.get("Length"), product.acquisition.length_lines),
        ("ExposureTime", parsed.get("ExposureTime"), product.acquisition.exposure_time),
        ("Gain", parsed.get("Gain"), product.acquisition.gain),
        ("InternalLineRate", float(line_rate_hz), product.acquisition.internal_line_rate_hz),
    ):
        if before is not None and before != after:
            note(f"設定值 {key}={_display(before)} 超出可用範圍，已調整為 {_display(after)}。")

    for key, before, after in (
        (
            "ExternalFrameTriggerOneFrame",
            parsed.get("ExternalFrameTriggerOneFrame"),
            product.trigger.external_frame_one_frame,
        ),
        (
            "ExternalFrameTriggerOneFrameCompareFromEncoder",
            parsed.get("ExternalFrameTriggerOneFrameCompareFromEncoder"),
            product.trigger.compare_follows_encoder,
        ),
        (
            "ExternalFrameTriggerOneFrameSetEncoderOnTrigger",
            parsed.get("ExternalFrameTriggerOneFrameSetEncoderOnTrigger"),
            product.trigger.set_encoder_on_trigger,
        ),
    ):
        if before is True and after is False:
            note(
                f"設定值 {key}=true 與觸發模式 {_display(product.trigger.mode)} 衝突，"
                "依 Trigger 互斥規則已忽略。"
            )

    cleared = [
        index
        for index, channel in enumerate(machine.meter_wheel.extension_channels)
        if meter_output_states >> index & 1 and not channel.output_state
    ]
    if cleared:
        note(
            "設定值 MeterWheelExtensionCompareOutputStates 的通道 "
            + "、".join(f"CMP{index}" for index in cleared)
            + " 同時開啟 MeterWheelExtensionCompareMask，依規則已清除其輸出狀態。"
        )

    if not seen_imported:
        note("設定檔沒有 VisionFlow 可匯入的設定，已全部使用預設值。")

    return ImportedCcdSettings(
        machine=machine,
        product=product,
        warnings=tuple(warnings),
        source=str(source),
    )


def import_ccd_settings_ini(path, environ: Mapping[str, str] | None = None) -> ImportedCcdSettings:
    """Read a `settings.ini` and convert it into machine-level and product-level CCD settings.

    `path` accepts a `str`, an `os.PathLike` or a directory holding `settings.ini`. An empty path
    falls back to `VISIONFLOW_CCD_SETTINGS_INI` from `environ` (default `os.environ`); an explicit
    path always wins. A missing or non-UTF-8 file raises :class:`CcdSettingsImportError`.
    """
    environment = os.environ if environ is None else environ
    candidate = str(path).strip() if path is not None else ""
    if not candidate:
        candidate = str(environment.get(SETTINGS_PATH_ENV, "") or "").strip()
    if not candidate:
        raise CcdSettingsImportError(
            f"沒有指定 CCD 設定檔路徑，請選擇 settings.ini 或以環境變數 {SETTINGS_PATH_ENV} 指定。"
        )

    target = Path(candidate)
    if target.is_dir():
        target = target / SETTINGS_FILE_NAME
    if not target.is_file():
        raise CcdSettingsImportError(f"找不到 CCD 設定檔：{target}")
    try:
        data = target.read_bytes()
    except OSError as exc:
        raise CcdSettingsImportError(f"CCD 設定檔無法讀取：{target}（{exc}）") from exc
    try:
        # The C# writer emits UTF-8 with a BOM; `utf-8-sig` also accepts a plain UTF-8 file.
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CcdSettingsImportError(f"CCD 設定檔不是 UTF-8 文字：{target}（{exc}）") from exc
    return parse_ccd_settings_ini(text, source=str(target))
