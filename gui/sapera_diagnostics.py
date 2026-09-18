from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from core.logging_system import LogMixin
from devices.ccd_models import (
    TRIGGER_MODE_LABELS,
    AcquisitionSettings,
    CameraConnectionSettings,
    CameraRecipeSettings,
    TriggerMode,
    TriggerSettings,
)
from devices.sapera_diagnose import (
    DIAGNOSE_LOG_SUBDIR,
    STEP_TITLES,
    DiagnoseReport,
    DiagnoseStep,
    run_sapera_diagnose,
)
from devices.sapera_api import SaperaError, SaperaVersions
from devices.sapera_camera import ApplyNote

# ============================================================
# CCD 頁的 Sapera 診斷支援（Todo.md P11）。
#
# 這個模組只放「診斷 worker」與「診斷匯出」兩件事，不含任何 Qt widget：
# `run_sapera_diagnose` 由 `devices/sapera_diagnose.py` 擁有且介面固定，
# 這裡只負責把它移到 worker thread 執行，並把綁定真正收集到的資料寫成一份可帶走的報告。
#
# 相機機台的檔案帶不出來，所以匯出報告不假裝涵蓋沒有收集的項目：Live Features 與
# Acq Params 列舉不在這個綁定的實作範圍內（見 `NOT_COLLECTED_NOTES`），報告與畫面都會明講。
# ============================================================

EXPORT_SCHEMA = "visionflow-sapera-diagnostics-export/v1"
EXPORT_FILE_PREFIX = "sapera-diagnostics-"

#: Which hardware facts this binding never collects. Stated verbatim in the export and on screen.
NOT_COLLECTED_NOTES = (
    "未收集：Live Features 列舉（SapAcqDevice 上每個 feature 的名稱、型別、存取模式與值）。"
    "此綁定只寫入 xx_ccd 已確認的 Exposure／Gain／Line Rate／Trigger 項目。",
    "未收集：Acq Params 完整列舉（SapAcquisition 每個 Prm 的可用性、能力值與目前值）。"
    "此綁定只讀寫 xx_ccd 已確認的 CROP_HEIGHT、觸發與線觸發相關參數。",
    "未收集：SapAcquisition 事件與 SignalNotify 的即時時序，以及相機端 feature 的完整 dump。",
)
NOT_COLLECTED_STATEMENT = "未收集：Live Features／Acq Params 列舉（本綁定不收集這兩個項目）。"

def not_collected_text() -> str:
    """One Traditional-Chinese statement naming what this export does not cover."""

    return NOT_COLLECTED_STATEMENT + "".join(NOT_COLLECTED_NOTES)


@dataclass(frozen=True)
class SaperaDiagnosticsReport:
    """Everything the CCD binding actually collected, ready to be written as one UTF-8 file."""

    connection: CameraConnectionSettings
    product: CameraRecipeSettings
    apply_notes: tuple[str, ...]
    versions: SaperaVersions
    versions_summary: str
    versions_mismatch: bool
    availability_reason: str
    missing_api_members: tuple[str, ...] = ()
    diagnose_lines: tuple[str, ...] = ()
    diagnose_summary: str = ""
    diagnose_report_path: str = ""
    diagnose_log_path: str = ""
    exported_at: str = ""

    def requested_lines(self) -> tuple[str, ...]:
        connection = self.connection
        acquisition: AcquisitionSettings = self.product.acquisition
        trigger: TriggerSettings = self.product.trigger
        return (
            f"擷取伺服器：{connection.server_name or '（未設定）'}",
            f"Resource Index：{connection.resource_index}",
            f"CCF 檔案：{connection.config_file_path or '（未設定）'}",
            f"相機功能伺服器：{connection.device_feature_server_name or '（未設定）'}"
            f"／Resource {connection.device_feature_resource_index}",
            f"曝光時間：{acquisition.exposure_time:g}",
            f"增益：{acquisition.gain:g}",
            f"影像長度（線）：{acquisition.length_lines}",
            f"內部線速率（Hz）：{acquisition.internal_line_rate_hz}",
            f"觸發模式：{TRIGGER_MODE_LABELS.get(TriggerMode(trigger.mode), trigger.mode.value)}"
            f"（{trigger.mode.value}）",
            f"外部觸發單張：{'是' if trigger.external_frame_one_frame else '否'}",
            f"自動寫入 Compare：{'是' if trigger.compare_follows_encoder else '否'}",
            f"同時寫入 Encoder：{'是' if trigger.set_encoder_on_trigger else '否'}",
            f"外部觸發後自動存圖：{'是' if self.product.auto_save_external_one_frame else '否'}",
            f"軟體觸發後自動存圖：{'是' if self.product.auto_save_software_trigger else '否'}",
        )

    def version_lines(self) -> tuple[str, ...]:
        versions = self.versions
        mismatch = "是（E-0203）" if self.versions_mismatch else "否"
        lines = [
            f"managed DLL 檔案版本：{versions.assembly_file_version or '未知'}",
            f"Sapera runtime（native）版本：{versions.native_file_version or '未知'}",
            f"managed 組件版本：{versions.assembly_version or '未知'}",
            f"managed 路徑：{versions.assembly_path or '未知'}",
            f"native 路徑：{versions.native_path or '未知'}",
            f"版本不一致：{mismatch}",
            f"版本摘要：{self.versions_summary}",
        ]
        if self.versions_mismatch:
            lines.append(f"Sapera runtime 版本不符：{self.versions_summary}")
        return tuple(lines)


def report_text(report: SaperaDiagnosticsReport) -> str:
    """Build the UTF-8 export body; every section states where its data came from."""

    lines = [
        "VisionFlow AOI Sapera 診斷匯出（CCD 控制頁）",
        f"匯出時間：{report.exported_at or '（未知）'}",
        f"schema：{EXPORT_SCHEMA}",
        "",
        "== 相機設定（要求值） ==",
        "來源：CCD 頁「Sapera 位置」與目前產品設定（ccd_machine.json 與已載入 Recipe）。",
    ]
    lines.extend(f"  {line}" for line in report.requested_lines())
    lines.append("")
    lines.append("== 上次連線的套用結果（apply_notes：寫入值與讀回值） ==")
    if report.apply_notes:
        lines.extend(f"  {line}" for line in report.apply_notes)
    else:
        lines.append("  （本次執行尚未連線，沒有套用紀錄）")
    lines.append("")
    lines.append("== Sapera 版本 ==")
    lines.extend(f"  {line}" for line in report.version_lines())
    lines.append("")
    lines.append("== 相機可用性 ==")
    lines.append(f"  {report.availability_reason or 'Sapera 載入正常，沒有錯誤碼。'}")
    lines.append("")
    lines.append("== API 自檢（SAPERA_API_MANIFEST） ==")
    if report.missing_api_members:
        lines.append(f"  缺少 {len(report.missing_api_members)} 個成員（E-0301）：")
        lines.extend(f"    {member}" for member in report.missing_api_members)
    else:
        lines.append("  沒有缺少的成員（自檢通過，或 Sapera 尚未載入而無法檢查）。")
    lines.append("")
    lines.append("== 最近一次現場診斷（S1–S8 短碼） ==")
    if report.diagnose_lines:
        lines.append(f"  總結：{report.diagnose_summary}")
        lines.extend(f"  {line}" for line in report.diagnose_lines)
        lines.append(f"  完整報告：{report.diagnose_report_path or '（未寫入）'}")
        lines.append(f"  機器可讀報告：{report.diagnose_log_path or '（未寫入）'}")
    else:
        lines.append("  本次執行尚未執行「執行相機診斷」。")
    lines.append("")
    lines.append("== 未收集 ==")
    lines.append(f"  {NOT_COLLECTED_STATEMENT}")
    lines.extend(f"  {note}" for note in NOT_COLLECTED_NOTES)
    lines.append("")
    lines.append("本檔留在機台；相機機台的檔案無法攜出，請只抄回上面的短碼與實際讀回值。")
    return "\n".join(lines) + "\n"


def _stamp(clock: Callable[[], datetime] | None, exported_at: str) -> str:
    if exported_at:
        return exported_at
    try:
        now = (clock or datetime.now)()
        if isinstance(now, datetime):
            return now.strftime("%Y%m%d-%H%M%S")
    except Exception:  # noqa: BLE001 - a broken clock must never fail the export
        pass
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def export_filename(clock: Callable[[], datetime] | None = None, exported_at: str = "") -> str:
    return f"{EXPORT_FILE_PREFIX}{_stamp(clock, exported_at)}.txt"


def write_diagnostics_report(
    report: SaperaDiagnosticsReport,
    *,
    log_dir: str | Path | None = None,
    clock: Callable[[], datetime] | None = None,
) -> Path:
    """Write the report as UTF-8 into `log_dir` (default: the diagnose runner's own folder).

    Raises ``OSError`` when the directory or file cannot be written; the caller turns that into an
    operator notice. Nothing is ever written outside `log_dir`.
    """

    directory = Path(log_dir) if log_dir is not None else Path(DIAGNOSE_LOG_SUBDIR)
    directory.mkdir(parents=True, exist_ok=True)
    exported_at = _stamp(clock, report.exported_at)
    path = directory / f"{EXPORT_FILE_PREFIX}{exported_at}.txt"
    path.write_text(report_text(_with_stamp(report, exported_at)), encoding="utf-8")
    return path


def _with_stamp(report: SaperaDiagnosticsReport, exported_at: str) -> SaperaDiagnosticsReport:
    return replace(report, exported_at=exported_at)


# ---- worker ------------------------------------------------------------------------------------

class SaperaDiagnoseWorker(QObject, LogMixin):
    """Run S1-S8 off the GUI thread; the report crosses threads as one `object` signal.

    S7 may wait several seconds for a frame, so the run must never happen on the GUI thread. The
    worker owns no Qt widgets and never raises: `run_sapera_diagnose` already turns every failure
    into a FAIL/SKIP step.
    """

    finished = Signal(object)

    def __init__(
        self,
        *,
        runner=None,
        connection: CameraConnectionSettings | None = None,
        acquisition: AcquisitionSettings | None = None,
        trigger: TriggerSettings | None = None,
        log_dir: str | Path | None = None,
    ):
        super().__init__()
        self._runner = runner or run_sapera_diagnose
        self._connection = connection or CameraConnectionSettings()
        self._acquisition = acquisition or AcquisitionSettings()
        self._trigger = trigger or TriggerSettings()
        self._log_dir = log_dir
        self.thread_id: int | None = None
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    @Slot()
    def run(self) -> None:
        import threading  # noqa: PLC0415 - only needed to report which thread executed the run

        self.thread_id = threading.get_ident()
        try:
            report = self._runner(
                connection=self._connection,
                acquisition=self._acquisition,
                trigger=self._trigger,
                log_dir=self._log_dir,
            )
        except Exception:  # noqa: BLE001 - the runner promises not to raise; a bug must still be shown
            self.logger.exception("Sapera GUI diagnosis failed unexpectedly")
            report = _failed_report()
        self.finished.emit(report)


def _failed_report() -> DiagnoseReport:
    """Only reached if `run_sapera_diagnose` itself raised, which its contract forbids."""

    error = SaperaError("E-0901", "診斷流程未預期結束")
    step = DiagnoseStep("S1", STEP_TITLES["S1"], "FAIL", f"S1 FAIL {error.code} {error.detail}")
    return DiagnoseReport((step,), "", "", "S1-S8：0 PASS、1 FAIL、0 SKIP", ())


def apply_note_lines(notes: Sequence[ApplyNote] | None) -> tuple[str, ...]:
    """`ApplyNote.line()` for every recorded write/readback of the last connect."""

    return tuple(note.line() for note in (notes or ()))
