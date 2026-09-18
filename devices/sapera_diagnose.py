from __future__ import annotations

import json
import logging
import re
import struct
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from devices.ccd_models import (
    AcquisitionSettings,
    CameraConnectionSettings,
    CameraState,
    DeviceError,
    TriggerSettings,
)
from devices.sapera_api import (
    DEFAULT_CCF_SUBDIR,
    DEFAULT_SAPERA_DIR,
    DLL_PATH_ENV,
    ERROR_MESSAGES,
    SAPERADIR_ENV,
    TARGET_SAPERA_VERSION,
    SaperaError,
    SaperaRuntime,
    SaperaVersions,
    dotnet_exception_name,
    load_runtime,
    locate_assembly,
    translate_exception,
)
from devices.sapera_camera import BUFFER_COUNT, SaperaLineScanCamera

# ============================================================
# Field diagnosis for Sapera LT (Todo.md P11).
#
# The camera machine is offline: nothing can be copied out of it, and only what an operator reads
# off the screen and writes down by hand comes back. Every step therefore produces one short line
# (`"<code> <STATUS> <error-code-or-detail>"`) built from the codes in `sapera_api.ERROR_MESSAGES`;
# docs/sapera-diagnose.md is the operator table for those codes.
#
# Steps run strictly S1 -> S8 and stop short of hardware when an earlier step failed: a step that
# cannot run is recorded as SKIP, never omitted. Sapera objects are only created in S5 and only
# after S1-S4 proved the install, the managed DLL, the API surface and the enumeration. Nothing in
# this module raises out of `run_sapera_diagnose`: an unexpected exception becomes a FAIL step.
# ============================================================

LOGGER = logging.getLogger(__name__)

DIAGNOSE_LOG_SUBDIR = Path("outputs") / "logs" / "camera"
DIAGNOSE_SCHEMA = "visionflow-sapera-diagnose/v1"
FRAME_WAIT_TIMEOUT_SEC = 5.0
FRAME_WAIT_POLL_SEC = 0.05
_SHORT_LINE_MAX = 60

# Every interop method S5 and S6 use. The recording proxy delegates exactly these names and turns
# each call into one report line; `sapera_api.SAPERA_API_MANIFEST` covers the .NET types those
# calls reach, and S3 verifies that manifest before any hardware access.
_MANAGED_METHODS = (
    "server_count",
    "server_name",
    "resource_count",
    "resource_name",
    "location",
    "new_acq_device",
    "feature_available",
    "feature_access_mode",
    "set_feature_string",
    "set_feature_int64",
    "get_feature_string",
    "update_features",
    "new_acquisition",
    "acq_param_available",
    "acq_get_int",
    "acq_set_int",
    "acq_set_val",
    "acq_capability",
    "acq_set_cc1",
    "acq_read_cc1",
    "acq_signal_present",
    "acq_enable_signal_notify",
    "new_buffers",
    "buffer_clear",
    "buffer_format",
    "buffer_read",
    "new_transfer",
    "grab",
    "snap",
    "freeze",
)
# A recording proxy must cover every method `SaperaLineScanCamera` calls; the manifest check in
# `sapera_api` owns the type-level members, this tuple owns the Python-level method names.
MANAGED_INTEROP_METHODS = _MANAGED_METHODS

STEP_TITLES = {
    "S1": "Sapera 安裝與版本",
    "S2": "載入 SapClassBasic.dll",
    "S3": "Sapera API 自檢",
    "S4": "列舉 server／resource／CCF",
    "S5": "建立並釋放 Sapera 物件",
    "S6": "連線並寫入參數後讀回",
    "S7": "Snap 一張並檢查影像",
    "S8": "斷線與清理",
}


# ---- public result types ----------------------------------------------------------------------

@dataclass(frozen=True)
class DiagnoseNote:
    """One line of the full report; `code` is set on failures and warnings."""

    code: str
    text: str

    def line(self) -> str:
        return f"{self.code} {self.text}" if self.code else self.text


@dataclass(frozen=True)
class DiagnoseStep:
    code: str        # "S1".."S8"
    title: str       # Traditional Chinese
    status: str      # "PASS" | "FAIL" | "SKIP"
    short: str       # one manually-copyable line, e.g. "S6 FAIL E-0602 Exposure 寫入失敗"
    details: tuple[DiagnoseNote, ...] = ()

    def line(self) -> str:
        return self.short


@dataclass(frozen=True)
class DiagnoseReport:
    steps: tuple[DiagnoseStep, ...]
    report_path: str
    log_path: str
    summary_text: str = ""
    sapera_calls: tuple[str, ...] = ()

    def lines(self) -> tuple[str, ...]:
        return tuple(step.line() for step in self.steps)

    def numeric_lines(self) -> tuple[str, ...]:
        """One all-digit code per step; the field writes these down instead of Chinese prose."""

        return tuple(numeric_code(step) for step in self.steps)

    def numeric_line(self) -> str:
        """All step codes on one line, space separated, for a single hand-copied row."""

        return " ".join(self.numeric_lines())

    def summary(self) -> str:
        return self.summary_text or _summary_text(self.steps)

    @property
    def passed(self) -> bool:
        return bool(self.steps) and all(step.status == "PASS" for step in self.steps)


_ERROR_CODE_RE = re.compile(r"E-(\d{4})")
# Numeric causes that are not error codes. The four digits of an `E-xxxx` code are used verbatim, so
# the whole field report is digits and the operator never has to transcribe Chinese.
_NUMERIC_PASS = "0000"
_NUMERIC_SKIP = "9999"
_NUMERIC_FAIL_NO_CODE = "9998"
_NUMERIC_LEGEND = {
    _NUMERIC_PASS: "PASS",
    _NUMERIC_SKIP: "SKIP（前一步失敗）",
    _NUMERIC_FAIL_NO_CODE: "FAIL，無錯誤碼（看報告檔）",
}


def numeric_code(step: DiagnoseStep) -> str:
    """`<step 2 digits><cause 4 digits>`: `060602` = S6 failed with E-0602, `010000` = S1 passed.

    The cause digits are the `E-xxxx` code without its `E-`, so the documented error-code table is
    also the decode table: 0000 PASS, 9999 SKIP, 9998 FAIL without a code.
    """

    digits = "".join(character for character in str(step.code) if character.isdigit())
    step_part = (digits or "0")[-2:].rjust(2, "0")
    if step.status == "PASS":
        cause = _NUMERIC_PASS
    elif step.status == "SKIP":
        cause = _NUMERIC_SKIP
    else:
        match = _ERROR_CODE_RE.search(str(step.short))
        cause = match.group(1) if match else _NUMERIC_FAIL_NO_CODE
    return f"{step_part}{cause}"


def numeric_legend() -> dict[str, str]:
    """The non-error-code causes, for docs and tests."""

    return dict(_NUMERIC_LEGEND)


def _summary_text(steps: tuple[DiagnoseStep, ...]) -> str:
    counts = {status: sum(1 for step in steps if step.status == status) for status in ("PASS", "FAIL", "SKIP")}
    scope = f"{steps[0].code}-{steps[-1].code}" if steps else "S1-S8"
    return f"{scope}：{counts['PASS']} PASS、{counts['FAIL']} FAIL、{counts['SKIP']} SKIP"


def _short(code: str, status: str, detail: str = "") -> str:
    """`"<code> <STATUS> <detail>"`, capped so it survives being written down by hand."""

    parts = [code, status] + ([str(detail).strip()] if detail and str(detail).strip() else [])
    line = " ".join(parts)
    if len(line) <= _SHORT_LINE_MAX:
        return line
    return line[: _SHORT_LINE_MAX - 1].rstrip() + "…"


def _badge(code: str, extra: str = "") -> str:
    """`"<code> <localized summary> [identifier]"`.

    The localized summary is the operator's primary reading and is never truncated. `extra` is a
    short identifier (a missing .NET member, a Sapera object name); it is added only when it fits
    the copyable budget, and Chinese prose is never appended because the summary already says it.
    The full wording always stays in the report file.
    """

    summary = ERROR_MESSAGES.get(code, "")
    base = f"{code} {summary}".strip()
    if not extra or not extra[0].isascii() or extra in base:
        return base
    line = f"{base} {extra}"
    return line if len(line) <= _SHORT_LINE_MAX else base


def _failure_detail(error: SaperaError, label: str = "") -> str:
    """Short operator text: the localized message first, then a trimmed form of the raw detail."""

    summary = ERROR_MESSAGES.get(error.code, "")
    if label and (label in summary or summary in label):
        label = ""  # the localized message already names this item
    head = " ".join(part for part in (label, summary) if part)
    text = str(error.detail or "").strip()
    if not text:
        return head or ""
    name = text.split(":", 1)[0].strip()
    if text.startswith(name) and name and all(character.isascii() for character in name):
        # A .NET type or exception name: the operator only needs the short class name.
        detail = name.rsplit(".", 1)[-1]
    else:
        # An operator explanation such as "SapAcqDevice.Create() 回傳 false": keep the first clause.
        detail = text.split("；", 1)[0].split("，", 1)[0].split("(", 1)[0].strip()
    return f"{head}：{detail}" if head else detail


# ---- the machine's own Sapera install ---------------------------------------------------------

def _env_value(environ, name: str) -> str:
    try:
        return str(environ.get(name) or "")
    except Exception:  # noqa: BLE001 - a hostile environ mapping must not break the diagnosis
        return ""


def _file_version_text(path: Path) -> str:
    """Read the PE fixed file version without .NET, so S1 needs no pythonnet (stdlib only).

    `VS_FIXEDFILEINFO` layout: signature(4), struct version(4), FileVersionMS(4), FileVersionLS(4),
    where the major version is the high word of MS and the minor version the low word of MS.
    """

    try:
        data = path.read_bytes()
        offset = data.find(b"\xbd\x04\xef\xfe")
        if offset < 0 or offset + 16 > len(data):
            return ""
        _low, _high, file_version_ms, _file_version_ls = struct.unpack_from("<IIII", data, offset)
        return f"{file_version_ms >> 16}.{file_version_ms & 0xFFFF}"
    except (OSError, ValueError, struct.error):
        return ""


def _find_install_root(dll_path: Path | None) -> Path | None:
    """Locate the Sapera install root that owns `dll_path` (…\\Components\\NET\\Bin\\SapClassBasic.dll)."""

    if dll_path is None:
        return None
    current = dll_path.parent
    for _ in range(6):
        try:
            if current.is_dir():
                return current
        except OSError:  # pragma: no cover - defensive
            return None
        parent = current.parent
        if parent == current:
            return None
        current = parent
    return None


def _default_ccf_dir(dll_path: Path | None, environ) -> Path | None:
    sapera_dir = _env_value(environ, SAPERADIR_ENV) or DEFAULT_SAPERA_DIR
    try:
        if Path(sapera_dir).is_dir():
            return Path(sapera_dir).joinpath(*DEFAULT_CCF_SUBDIR)
    except OSError:  # pragma: no cover - defensive
        pass
    root = _find_install_root(dll_path)
    if root is None:
        return None
    return root.joinpath(*DEFAULT_CCF_SUBDIR)


def _ccf_files(ccf_dir: Path | None) -> list[Path]:
    if ccf_dir is None:
        return []
    try:
        if not ccf_dir.is_dir():
            return []
        return sorted(path for path in ccf_dir.glob("*.ccf") if path.is_file())
    except OSError:  # pragma: no cover - defensive
        return []


def _resolve_ccf(configured: str, ccf_dir: Path | None) -> tuple[str, tuple[Path, ...], str]:
    """Use the configured CCF when it exists, otherwise the only CCF under Sapera CamFiles\\User."""

    if configured:
        try:
            if Path(configured).is_file():
                return configured, _ccf_files(ccf_dir), ""
            return "", _ccf_files(ccf_dir), f"設定的 CCF 不存在：{configured}"
        except OSError:  # pragma: no cover - defensive
            return "", _ccf_files(ccf_dir), f"設定的 CCF 無法讀取：{configured}"
    files = _ccf_files(ccf_dir)
    if len(files) == 1:
        return str(files[0]), files, ""
    if not files:
        return "", (), "找不到 CCF 檔"
    return "", files, f"CamFiles\\User 有多個 CCF（{len(files)} 個），請指定 --sapera-ccf"


# ---- recorded Sapera calls and device logs -----------------------------------------------------

class _RecordingInterop:
    """Records one human-readable line per Sapera call, then delegates to the real interop."""

    def __init__(self, inner, record: Callable[[str], None]):
        self._inner = inner
        self._record = record
        for name in _MANAGED_METHODS + ("create", "destroy", "dispose"):
            if callable(getattr(inner, name, None)):
                setattr(self, name, self._method(name))
        # `initialized` is a pure read of the object in the real interop; nothing to record.
        self.initialized = getattr(inner, "initialized")

    def _method(self, name: str):
        def call(*args, **kwargs):
            result = getattr(self._inner, name)(*args, **kwargs)
            self._record(f"{name}{_describe_args(args)} -> {_describe_result(result)}")
            return result

        return call


def _describe_args(args: tuple) -> str:
    if not args:
        return "()"
    return "(" + ", ".join(_describe_value(value) for value in args) + ")"


def _describe_value(value) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return repr(value)
    label = getattr(value, "kind", None)
    if label is not None:
        return f"<{label}>"
    return f"<{type(value).__name__}>"


def _describe_result(value) -> str:
    if hasattr(value, "kind"):
        return f"<{value.kind}>"
    if isinstance(value, (list, tuple)):
        return _describe_args(tuple(value))
    return _describe_value(value)


class _DeviceLogHandler(logging.Handler):
    """Attaches to the `devices` loggers so the report keeps every write/readback the camera logs."""

    def __init__(self, lines: list[str]):
        super().__init__(level=logging.DEBUG)
        self._lines = lines

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._lines.append(f"{record.levelname} {record.name}: {record.getMessage()}")
        except Exception:  # noqa: BLE001 - a broken record must never break the diagnosis
            pass


class _Context:
    """One diagnosis run: accumulated report lines, recorded Sapera calls and step notes."""

    def __init__(self):
        self.lines: list[str] = []
        self.calls: list[str] = []
        self.notes: list[DiagnoseNote] = []

    def add(self, text: str) -> None:
        """One detail line that carries no error code."""

        self.lines.append(text)

    def add_note(self, code: str, text: str) -> None:
        """One failure or warning line; `code` is an `ERROR_MESSAGES` key (or "" for plain notes)."""

        note = DiagnoseNote(code, text)
        self.notes.append(note)
        self.lines.append(note.line())

    def record_call(self, text: str) -> None:
        self.calls.append(text)


@dataclass
class _RuntimeState:
    runtime: SaperaRuntime | None = None
    interop: object | None = None
    assembly_path: Path | None = None
    ccf_dir: Path | None = None
    versions: SaperaVersions = field(default_factory=SaperaVersions)


@dataclass
class _ObjectState:
    created: list[tuple[object, str]] = field(default_factory=list)
    extra: list[tuple[object, str]] = field(default_factory=list)


# ---- steps -----------------------------------------------------------------------------------

def _step_s1(context: _Context, state: _RuntimeState, environ) -> tuple[str, str]:
    """S1: the machine's Sapera install and version, without .NET (pythonnet is S2's problem)."""

    configured = _env_value(environ, DLL_PATH_ENV)
    search = locate_assembly(environ=environ)
    state.assembly_path = Path(search.chosen) if search.chosen else None
    context.add(f"檢查路徑：{'；'.join(search.checked) or '（無）'}")
    if configured:
        context.add(f"{DLL_PATH_ENV}={configured}")

    if configured:
        if state.assembly_path is None or not state.assembly_path.is_file():
            context.add_note("E-0201", f"{DLL_PATH_ENV} 指定的 DLL 不存在或不是檔案：{configured}")
            return "FAIL", _badge("E-0201", "指定的 DLL 不存在")
        version = _file_version_text(state.assembly_path)
    elif search.sapera_dir is None:
        context.add_note("E-0104", f"未偵測到 Sapera 安裝目錄（{SAPERADIR_ENV} 或 {DEFAULT_SAPERA_DIR}）")
        return "FAIL", _badge("E-0104", "找不到安裝目錄")
    elif state.assembly_path is not None:
        version = _file_version_text(state.assembly_path)
    else:
        version = ""
    version = version or "未知"
    context.add(f"managed 檔案版本：{version}（目標 {TARGET_SAPERA_VERSION}）")
    if version != "未知" and not version.startswith("8.60"):
        context.add_note("E-0203", f"managed 檔案版本 {version} 非 8.60")
    return "PASS", f"Sapera {version}"


def _step_s2(context: _Context, state: _RuntimeState, environ, runtime_loader) -> tuple[str, str]:
    """S2: load the machine's own SapClassBasic.dll through the loader the camera will use."""

    try:
        runtime = runtime_loader()
    except SaperaError as exc:
        context.add_note(exc.code, f"{ERROR_MESSAGES.get(exc.code, 'Sapera 載入失敗')}：{exc.detail}")
        return "FAIL", _badge(exc.code, exc.detail)
    except Exception as exc:  # noqa: BLE001 - any load failure becomes an operator code
        error = translate_exception(exc, "E-0202")
        context.add_note(error.code, error.detail)
        return "FAIL", _badge(error.code, error.detail)

    if runtime is None:
        context.add_note("E-0202", "runtime_loader 未回傳 Sapera runtime")
        return "FAIL", _badge("E-0202")
    state.runtime = runtime
    versions = getattr(runtime, "versions", None) or SaperaVersions()
    context.add(f"managed：{versions.assembly_path or '未知'}")
    context.add(f"版本：{versions.summary()}")
    context.add(f"native runtime：{versions.native_path or '未知'}")
    if versions.mismatch:
        context.add_note("E-0203", f"managed／native 版本不一致：{versions.summary()}")
    return "PASS", f"managed {versions.assembly_file_version or versions.assembly_version or '未知'}／runtime {versions.native_file_version or '未知'}"


def _step_s3(context: _Context, state: _RuntimeState) -> tuple[str, str]:
    """S3: verify every .NET member the camera uses by reflection, before any hardware access."""

    try:
        missing = tuple(state.runtime.check_api())
    except Exception as exc:  # noqa: BLE001 - reflection failure is still an API failure
        error = translate_exception(exc, "E-0301")
        context.add_note(error.code, error.detail or dotnet_exception_name(exc))
        return "FAIL", _badge(error.code, dotnet_exception_name(exc))
    if missing:
        preview = "、".join(missing[:8]) + ("…" if len(missing) > 8 else "")
        context.add_note("E-0301", f"缺少 {len(missing)} 個成員：{preview}")
        return "FAIL", _badge("E-0301", missing[0])
    try:
        interop = state.runtime.interop()
    except Exception as exc:  # noqa: BLE001
        error = translate_exception(exc, "E-0301")
        context.add_note(error.code, error.detail)
        return "FAIL", _badge(error.code, dotnet_exception_name(exc))
    state.interop = _RecordingInterop(interop, context.record_call)
    return "PASS", "API 成員齊全"


def _step_s4(context: _Context, connection: CameraConnectionSettings, state: _RuntimeState) -> tuple[str, str]:
    """S4: enumerate servers, their Acq/AcqDevice resources, and the CCF files on the machine."""

    interop = state.interop
    try:
        server_count = int(interop.server_count())
    except Exception as exc:  # noqa: BLE001
        error = translate_exception(exc, "E-0401")
        context.add_note(error.code, error.detail)
        return "FAIL", _failure_detail(error)

    servers: list[str] = []
    for index in range(server_count):
        try:
            name = str(interop.server_name(index))
        except Exception as exc:  # noqa: BLE001
            error = translate_exception(exc, "E-0401")
            context.add_note(error.code, error.detail)
            return "FAIL", _failure_detail(error)
        servers.append(name)
        for kind in ("Acq", "AcqDevice"):
            try:
                count = int(interop.resource_count(name, kind))
            except Exception as exc:  # noqa: BLE001 - one server must not hide the others
                context.add_note(translate_exception(exc, "E-0401").code, f"{name} 的 {kind} 數量讀取失敗")
                continue
            names = []
            for resource_index in range(count):
                try:
                    names.append(str(interop.resource_name(name, kind, resource_index)))
                except Exception as exc:  # noqa: BLE001
                    context.add_note(translate_exception(exc, "E-0401").code, f"{name} 的 {kind} #{resource_index} 名稱讀取失敗")
            context.add(f"server {name}：{kind} {count} 個" + (f"（{'、'.join(names)}）" if names else ""))
    if not servers:
        context.add_note("E-0402", "Sapera 回報 0 個 server，請確認擷取卡驅動與 Sapera LT")

    ccf_path, ccf_found, ccf_problem = _resolve_ccf(connection.config_file_path, state.ccf_dir)
    listed = "、".join(path.name for path in ccf_found[:8]) + ("…" if len(ccf_found) > 8 else "")
    context.add(f"CCF 目錄：{state.ccf_dir or '未知'}")
    context.add(f"CCF 檔 {len(ccf_found)} 個" + (f"（{listed}）" if ccf_found else ""))
    context.add(f"本次使用 CCF：{ccf_path or '（未決定）'}")
    if ccf_problem:
        context.add_note("E-0403", ccf_problem)
    detail = f"{len(servers)} 個 server、CCF {len(ccf_found)} 個"
    if ccf_path:
        detail += f"（{Path(ccf_path).name}）"
    return "PASS", detail


def _step_s5(context: _Context, connection: CameraConnectionSettings, state: _RuntimeState, objects: _ObjectState) -> tuple[str, str]:
    """S5: create and release every Sapera object the camera uses, in the camera's own order."""

    interop = state.interop
    location = interop.location(connection.server_name, connection.resource_index)
    context.add(f"位置：{connection.server_name}#{connection.resource_index}")

    device = None
    try:
        device = interop.new_acq_device(location)
        if not interop.create(device):
            context.add_note("E-0501", "SapAcqDevice.Create() 回傳 false")
        else:
            objects.extra.append((device, "SapAcqDevice"))
    except Exception as exc:  # noqa: BLE001 - one failing object must not abort the others
        context.add_note(translate_exception(exc, "E-0501").code, f"SapAcqDevice 建立失敗：{dotnet_exception_name(exc)}")

    acquisition = None
    try:
        acquisition = interop.new_acquisition(location, connection.config_file_path, _ignore_event, _ignore_signal)
        if not interop.create(acquisition):
            context.add_note("E-0502", "SapAcquisition.Create() 回傳 false")
        else:
            objects.created.append((acquisition, "SapAcquisition"))
    except Exception as exc:  # noqa: BLE001
        context.add_note(translate_exception(exc, "E-0502").code, f"SapAcquisition 建立失敗：{dotnet_exception_name(exc)}")

    buffers = None
    memory = ""
    if acquisition is not None:
        try:
            buffers, memory = interop.new_buffers(acquisition, location, BUFFER_COUNT)
            if not interop.create(buffers):
                context.add_note("E-0503", "SapBufferWithTrash.Create() 回傳 false")
            else:
                objects.created.append((buffers, "SapBufferWithTrash"))
        except Exception as exc:  # noqa: BLE001
            context.add_note(translate_exception(exc, "E-0503").code, f"SapBufferWithTrash 建立失敗：{dotnet_exception_name(exc)}")

    transfer = None
    if acquisition is not None and buffers is not None:
        try:
            transfer = interop.new_transfer(acquisition, buffers, _ignore_frame)
            if not interop.create(transfer):
                context.add_note("E-0504", "SapAcqToBuf.Create() 回傳 false")
            else:
                objects.created.append((transfer, "SapAcqToBuf"))
        except Exception as exc:  # noqa: BLE001
            context.add_note(translate_exception(exc, "E-0504").code, f"SapAcqToBuf 建立失敗：{dotnet_exception_name(exc)}")

    context.add(f"已建立物件：{len(objects.created)} 個" + (f"（{'、'.join(label for _obj, label in objects.created)}）" if objects.created else ""))
    if memory:
        context.add(f"buffer 記憶體類型：{memory}")

    cleanup = _release(context, interop, objects.created + objects.extra)
    if cleanup:
        return "FAIL", _badge(cleanup[0].code, cleanup[0].label.split()[0])
    if objects.created:
        return "PASS", f"建立並釋放 {len(objects.created)} 個物件"
    return "PASS", "無物件可建立（已回報個別失敗）"


@dataclass(frozen=True)
class _ReleaseFailure:
    label: str
    code: str


def _release(context: _Context, interop, entries) -> list[_ReleaseFailure]:
    """Destroy then dispose newest-first, exactly like `SaperaLineScanCamera._cleanup`."""

    failures: list[_ReleaseFailure] = []
    for obj, label in reversed(entries):
        failures.extend(_guarded_call(context, lambda obj=obj: interop.initialized(obj) and interop.destroy(obj), label))
    for obj, label in reversed(entries):
        failures.extend(_guarded_call(context, lambda obj=obj: interop.dispose(obj), label))
    return failures


def _guarded_call(context: _Context, action, label: str) -> list[_ReleaseFailure]:
    """Both the real camera and this module keep going after a cleanup failure; report each one.

    `label` is the short copyable form (`"SapAcqToBuf dispose"`); the full report line names the
    object again so the two readings agree.
    """

    try:
        action()
    except Exception as exc:  # noqa: BLE001 - cleanup failures must never raise out of the run
        context.add_note("E-0801", f"{label.split()[0]}：{label} 失敗（{dotnet_exception_name(exc)}）")
        return [_ReleaseFailure(label, "E-0801")]
    return []


def _step_s6(
    context: _Context,
    camera: SaperaLineScanCamera,
    connection: CameraConnectionSettings,
    acquisition: AcquisitionSettings,
    trigger: TriggerSettings,
    state: _RuntimeState,
) -> tuple[str, str]:
    """S6: connect through `SaperaLineScanCamera`, then report the readbacks and failed writes."""

    if state.interop is None:
        context.add_note("E-0901", "沒有 Sapera interop 可用（S3 未完成）")
        return "FAIL", _badge("E-0901")
    try:
        status = camera.connect(connection, acquisition, trigger)
    except (SaperaError, DeviceError) as exc:
        code = getattr(exc, "code", "") or "E-0901"
        context.add_note(code, str(exc))
        return "FAIL", _failure_detail(exc if isinstance(exc, SaperaError) else SaperaError(code, str(exc)))

    notes = tuple(camera.apply_notes())
    context.add(f"連線狀態：{status.state.value}、{status.frame_width}×{status.frame_height}、訊號 {'有' if status.has_signal else '無'}")
    for note in notes:
        context.add(f"[{note.item}] {note.line()}")
    context.add("板卡參數讀回：" + _readback(acquisition))
    failures = [note for note in notes if note.code]
    if failures:
        first = failures[0]
        # The localized message already names the parameter, e.g. "E-0602 Exposure 寫入失敗".
        summary = ERROR_MESSAGES.get(first.code, "")
        extra = "" if first.item in summary else first.item
        return "FAIL", _badge(first.code, extra)
    return "PASS", f"參數寫入並讀回 {len(notes)} 項"


def _readback(acquisition: AcquisitionSettings) -> str:
    return (
        f"Exposure={int(acquisition.exposure_time)}、Gain={int(acquisition.gain)}、"
        f"Length={acquisition.length_lines}、LineRate={acquisition.internal_line_rate_hz}Hz"
    )


def _step_s7(context: _Context, camera: SaperaLineScanCamera, wait: Callable[[], None]) -> tuple[str, str]:
    """S7: one frame through Snap, then its size and grey statistics, within a bounded wait."""

    try:
        start_status = camera.status()
        camera.capture_frame()
        context.add(f"Snap 已啟動；state={start_status.state.value}")
    except (SaperaError, DeviceError) as exc:
        code = getattr(exc, "code", "") or "E-0701"
        context.add_note(code, str(exc))
        return "FAIL", _failure_detail(exc if isinstance(exc, SaperaError) else SaperaError(code, str(exc)))

    wait()
    frame = camera.latest_frame()
    if frame is None:
        detail = camera.status().message or "尚未收到影像"
        context.add_note("E-0702", f"{detail}（等待上限 {FRAME_WAIT_TIMEOUT_SEC:g} 秒）")
        return "FAIL", _badge("E-0702")
    if frame.size == 0:
        context.add_note("E-0703", "收到空影像")
        return "FAIL", _badge("E-0703")

    height, width = int(frame.shape[0]), int(frame.shape[1])
    values = frame.astype(np.float64)
    minimum, maximum, mean = float(values.min()), float(values.max()), float(values.mean())
    context.add(f"影像：{width}×{height}、dtype {frame.dtype}")
    context.add(f"灰階統計：min {minimum:.0f}、max {maximum:.0f}、mean {mean:.2f}")
    if int(frame.dtype.itemsize) != 1:
        context.add_note("E-0704", f"像素格式 {frame.dtype} 非 8-bit 單色")
    return "PASS", f"{width}×{height} min{minimum:.0f} max{maximum:.0f} mean{mean:.1f}"


def _step_s8(
    context: _Context,
    camera: SaperaLineScanCamera,
    objects: _ObjectState,
    state: _RuntimeState,
    note_from: int,
) -> tuple[str, str]:
    """S8: disconnect, then release anything S5 left behind; cleanup still reports PASS/FAIL."""

    message = ""
    try:
        camera.disconnect()
        message = camera.status().message
    except Exception as exc:  # noqa: BLE001 - cleanup must not raise out of the run
        context.add_note("E-0801", f"disconnect 失敗：{dotnet_exception_name(exc)}")
    if "E-0801" in message:
        context.add_note("E-0801", f"相機清理失敗：{message}")
    context.add(f"斷線後訊息：{message or '（無）'}")

    interop = getattr(camera, "_interop", None) or state.interop
    leftovers = _release(context, interop, objects.extra)
    # `_step_s5` already released its objects; only failures recorded during S8 count here.
    own = [note for note in context.notes[note_from:] if note.code == "E-0801"]
    if own or leftovers:
        # The localized summary already says the object could not be released; which object it was
        # is in the full report right above, so the short line stays copyable.
        return "FAIL", _badge("E-0801")
    context.add("Sapera 物件已全部釋放")
    return "PASS", "已斷線並清理"


def _ignore_frame(_trash: bool) -> None:
    """S5 owns no frame callback: the buffer is created and released, never transferred."""


def _ignore_event(_name: str) -> None:
    """S5 only proves the object can be created; acquisition events are S7's business."""


def _ignore_signal(_present: bool) -> None:
    """S5 only proves the object can be created; signal events are S6's business."""


# ---- report files ------------------------------------------------------------------------------

def _stamp(clock) -> str:
    try:
        now = clock()
        if isinstance(now, datetime):
            return now.strftime("%Y%m%d-%H%M%S")
    except Exception:  # noqa: BLE001 - a diagnostic clock must never fail the run
        LOGGER.debug("diagnose clock failed", exc_info=True)
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _steps_payload(steps: tuple[DiagnoseStep, ...]) -> list[dict]:
    return [
        {
            "code": step.code,
            "title": step.title,
            "status": step.status,
            "numeric": numeric_code(step),
            "short": step.short,
            "details": [note.line() for note in step.details],
        }
        for step in steps
    ]


def _text_report(step_payload: list[dict], context: _Context, versions: SaperaVersions, stamp: str, summary: str) -> str:
    lines = [
        "VisionFlow AOI Sapera 現場診斷報告",
        f"時間：{stamp}",
        f"總結：{summary}",
        f"managed：{versions.assembly_file_version or versions.assembly_version or '未知'}",
        f"native runtime：{versions.native_file_version or '未知'}",
        f"managed 路徑：{versions.assembly_path or '未知'}",
        f"native 路徑：{versions.native_path or '未知'}",
        "",
        "== 數字短碼（優先抄這一組） ==",
        "  格式：<步驟 2 位><原因 4 位>；原因＝錯誤碼去掉 E-（0000 PASS、9999 SKIP、9998 FAIL 無碼）",
        "  " + " ".join(entry.get("numeric", "") for entry in step_payload),
    ]
    lines.extend(f"  第 {entry['code']} 步：{entry.get('numeric', '')}" for entry in step_payload)
    lines.append("")
    lines.append("== 短碼（可人工抄回） ==")
    lines.extend(f"  {entry['short']}" for entry in step_payload)
    lines.append("")
    lines.append("== 步驟細節 ==")
    for entry in step_payload:
        lines.append(f"[{entry['code']}] {entry['title']}：{entry['status']}")
        lines.extend(f"    {detail}" for detail in entry["details"])
    lines.append("")
    lines.append("== 診斷過程 log ==")
    lines.extend(f"  {line}" for line in context.lines)
    lines.append("")
    lines.append("== Sapera 呼叫 log ==")
    lines.extend(f"  {line}" for line in context.calls)
    lines.append("")
    lines.append("本檔留在機台，無法攜出；請只抄回上面的短碼。")
    return "\n".join(lines) + "\n"


def _write_reports(log_dir: Path, stamp: str, text: str, payload: dict) -> tuple[str, str, str]:
    report_path = log_dir / f"sapera-diagnose-{stamp}.txt"
    log_path = log_dir / f"sapera-diagnose-{stamp}.json"
    problem = ""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return "", "", f"無法建立 log 目錄 {log_dir}：{exc}"
    try:
        report_path.write_text(text, encoding="utf-8")
    except OSError as exc:
        problem = f"無法寫入報告 {report_path}：{exc}"
    try:
        log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        problem = (problem + "；" if problem else "") + f"無法寫入 JSON {log_path}：{exc}"
    return str(report_path) if report_path.is_file() else "", str(log_path) if log_path.is_file() else "", problem


# ---- runner ------------------------------------------------------------------------------------

def _resolve_loader(runtime_loader, environ):
    if runtime_loader is None:
        return lambda: load_runtime(environ=environ)
    return runtime_loader


def _wait_for_frame(camera: SaperaLineScanCamera, clock) -> None:
    """Bounded wait: uses the injected clock when it can be, and always gives up after a few seconds."""

    deadline = time.monotonic() + FRAME_WAIT_TIMEOUT_SEC
    while camera.latest_frame() is None:
        if time.monotonic() >= deadline:
            return
        if clock is not None:
            try:
                clock()  # let an injected test clock advance its simulated elapsed time
            except Exception:  # noqa: BLE001
                pass
        time.sleep(FRAME_WAIT_POLL_SEC)


def _skip(context: _Context, code: str, reason: str, note_from: int, results: list[DiagnoseStep]) -> None:
    context.add(f"[{code}] {STEP_TITLES[code]}：SKIP（{reason}）")
    details = tuple(context.notes[note_from:])
    results.append(DiagnoseStep(code, STEP_TITLES[code], "SKIP", _short(code, "SKIP", reason), details))


def _record(context: _Context, code: str, status: str, detail: str, note_from: int, results: list[DiagnoseStep]) -> None:
    context.add(f"[{code}] {STEP_TITLES[code]}：{status}")
    details = tuple(context.notes[note_from:])
    results.append(DiagnoseStep(code, STEP_TITLES[code], status, _short(code, status, detail), details))


def run_sapera_diagnose(
    *,
    runtime_loader=None,
    environ: Mapping[str, str] | None = None,
    connection: CameraConnectionSettings | None = None,
    acquisition: AcquisitionSettings | None = None,
    trigger: TriggerSettings | None = None,
    log_dir=None,
    camera: SaperaLineScanCamera | None = None,
    clock: Callable[[], datetime] | None = None,
) -> DiagnoseReport:
    """Run S1-S8 and always return a report; this function never raises."""

    env = environ if environ is not None else _default_environ()
    settings = (connection or CameraConnectionSettings()).normalized()
    acquisition = (acquisition or AcquisitionSettings()).normalized()
    trigger = (trigger or TriggerSettings()).normalized()
    directory = Path(log_dir) if log_dir is not None else DIAGNOSE_LOG_SUBDIR
    stamp = _stamp(clock or datetime.now)
    loader = _resolve_loader(runtime_loader, env)

    context = _Context()
    state = _RuntimeState()
    objects = _ObjectState()
    results: list[DiagnoseStep] = []
    camera = camera if camera is not None else SaperaLineScanCamera(loader, clock=clock or time.monotonic)
    session_clock = getattr(camera, "_clock", None)
    handler = _DeviceLogHandler(context.lines)
    devices_logger = logging.getLogger("devices")
    devices_logger.addHandler(handler)
    connected = False
    try:
        note_from = len(context.notes)
        status, short = _run_guarded(context, "S1", lambda: _step_s1(context, state, env))
        _record(context, "S1", status, short, note_from, results)
        # S1 knows the install root, so the CCF directory is resolvable from here on.
        state.ccf_dir = _default_ccf_dir(state.assembly_path, env)

        for code, action in (
            ("S2", lambda: _step_s2(context, state, env, loader)),
            ("S3", lambda: _step_s3(context, state)),
            ("S4", lambda: _step_s4(context, settings, state)),
            ("S5", lambda: _step_s5(context, settings, state, objects)),
            ("S6", lambda: _step_s6(context, camera, settings, acquisition, trigger, state)),
            ("S7", lambda: _step_s7(context, camera, lambda: _wait_for_frame(camera, session_clock))),
        ):
            note_from = len(context.notes)
            if status != "PASS":
                _skip(context, code, "前一步失敗", note_from, results)
                continue
            status, short = _run_guarded(context, code, action)
            _record(context, code, status, short, note_from, results)
            if code == "S6":
                connected = camera.status().state != CameraState.OFFLINE

        note_from = len(context.notes)
        if connected:
            status, short = _run_guarded(context, "S8", lambda: _step_s8(context, camera, objects, state, note_from))
            _record(context, "S8", status, short, note_from, results)
        else:
            _skip(context, "S8", "S6 未連線", note_from, results)
    except Exception as exc:  # noqa: BLE001 - nothing may escape `run_sapera_diagnose`
        LOGGER.exception("Sapera diagnosis failed unexpectedly")
        error = translate_exception(exc, "E-0901")
        note_from = len(context.notes)
        context.add_note(error.code, f"未預期錯誤：{error.detail}")
        _record(context, "S8", "FAIL", _badge(error.code, error.detail.split(":", 1)[0]), note_from, results)
    finally:
        devices_logger.removeHandler(handler)
        try:
            camera.close()
        except Exception:  # noqa: BLE001 - best effort only; S8 already reported the real cleanup
            LOGGER.debug("closing the diagnostic camera failed", exc_info=True)

    versions = state.versions if state.runtime is None else state.runtime.versions
    steps = tuple(results)
    summary = _summary_text(steps)
    payload = {
        "schema": DIAGNOSE_SCHEMA,
        "timestamp": stamp,
        "summary": summary,
        "numeric": " ".join(numeric_code(step) for step in steps),
        "passed": bool(steps) and all(step.status == "PASS" for step in steps),
        "versions": {
            "assembly_path": versions.assembly_path,
            "assembly_version": versions.assembly_version,
            "assembly_file_version": versions.assembly_file_version,
            "native_path": versions.native_path,
            "native_file_version": versions.native_file_version,
        },
        "steps": _steps_payload(steps),
        "sapera_calls": list(context.calls),
        "log": list(context.lines),
    }
    report_path, log_path, problem = _write_reports(
        directory, stamp, _text_report(_steps_payload(steps), context, versions, stamp, summary), payload
    )
    if problem:
        context.add_note("E-0901", problem)
    return DiagnoseReport(steps, report_path, log_path, summary, tuple(context.calls))


def _run_guarded(context: _Context, code: str, action) -> tuple[str, str]:
    """One step's body: any unexpected exception becomes a FAIL with an operator code."""

    try:
        return action()
    except SaperaError as exc:
        context.add_note(exc.code, f"{code}：{exc.detail}")
        return "FAIL", _badge(exc.code, exc.detail)
    except Exception as exc:  # noqa: BLE001
        error = translate_exception(exc, "E-0901")
        context.add_note(error.code, f"{code} 未預期錯誤：{error.detail}")
        return "FAIL", _badge(error.code, error.detail)


def _default_environ() -> Mapping[str, str]:
    import os  # noqa: PLC0415 - only needed for the default path

    return os.environ
