from __future__ import annotations

import datetime
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QImage

from core.camera_monitor_processor import CameraFrameQueue, CapturedFrame, RawFrameSaver
from core.logging_system import LogMixin
from devices.ccd_models import (
    CAMERA_STATE_LABELS,
    AcquisitionSettings,
    CameraConnectionSettings,
    CameraRecipeSettings,
    CameraState,
    CameraStatus,
    CcdMachineSettings,
    DeviceAvailability,
    DeviceError,
    ExtensionCompareChannel,
    ImageSaveFormat,
    MeterWheelSettings,
    MeterWheelSnapshot,
    MultipleRate,
    SaveSettings,
    TriggerMode,
    TriggerSettings,
)
from devices.ccd_settings_store import CcdMachineSettingsStore
from devices.factory import CcdDevices
from devices.frame_writer import SaveQueueStats, SnapshotSaveQueue, write_frame_atomic
from devices.trigger_automation import (
    SOFTWARE_TRIGGER_POLL_SEC,
    AutoSaveRequests,
    ExternalTriggerActions,
    SoftwareTriggerMonitor,
    external_trigger_actions,
    software_frame_requests_auto_save,
)

PREVIEW_MAX_DIMENSION = 2048
METER_WHEEL_POLL_MS = 200
DEFAULT_SNAPSHOT_DIR = Path("outputs") / "ccd_snapshots"


def preview_qimage(frame: np.ndarray, max_dimension: int = PREVIEW_MAX_DIMENSION) -> QImage:
    """Downscale a full-resolution frame for display; the source frame is never modified."""
    height, width = frame.shape[:2]
    scale = min(1.0, float(max_dimension) / max(height, width))
    preview = frame
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        preview = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    if preview.ndim == 3:
        preview = np.ascontiguousarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
        image_format = QImage.Format.Format_RGB888
    else:
        preview = np.ascontiguousarray(preview)
        image_format = QImage.Format.Format_Grayscale8
    image = QImage(preview.data, preview.shape[1], preview.shape[0], preview.strides[0], image_format)
    return image.copy()


class PreviewFrameConverter:
    """One background thread that converts only the newest frame; older pending frames are dropped."""

    def __init__(self, on_image: Callable[[QImage, int, int], None], max_dimension: int = PREVIEW_MAX_DIMENSION):
        self._on_image = on_image
        self._max_dimension = max_dimension
        self._condition = threading.Condition()
        self._pending: np.ndarray | None = None
        self._closed = False
        self._thread: threading.Thread | None = None
        self.dropped_frames = 0

    def submit(self, frame: np.ndarray) -> None:
        with self._condition:
            if self._closed:
                return
            if self._pending is not None:
                self.dropped_frames += 1
            self._pending = frame
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="ccd-preview", daemon=True)
                self._thread.start()
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                frame, self._pending = self._pending, None
            height, width = frame.shape[:2]
            self._on_image(preview_qimage(frame, self._max_dimension), width, height)


@dataclass(frozen=True)
class CcdCameraSettingsView:
    connection: CameraConnectionSettings
    product: CameraRecipeSettings
    save: SaveSettings
    pending_hardware_write: bool
    source_text: str = ""


class CcdController(QObject, LogMixin):
    """Owns the CCD camera and meter-wheel sessions for the whole application window.

    Leaving the CCD screen never disconnects hardware; `close()` does. Camera settings are
    written by `connect_camera()` only, matching the Sapera offline-apply workflow.
    """

    camera_status_changed = Signal(object)
    camera_settings_changed = Signal(object)
    product_settings_applied = Signal(object)
    meter_wheel_changed = Signal(object)
    meter_wheel_settings_changed = Signal(object)
    preview_image_ready = Signal(QImage, int, int)
    save_stats_changed = Signal(object)
    software_trigger_monitor_changed = Signal(bool)
    status_message = Signal(str)
    notice = Signal(str, str)
    _frame_arrived = Signal()
    _external_trigger_arrived = Signal(object)
    _software_capture_requested = Signal(int, int)
    _software_monitor_failed = Signal(str)
    _auto_save_rejected = Signal()

    software_trigger_poll_sec = SOFTWARE_TRIGGER_POLL_SEC

    def __init__(self, devices: CcdDevices, store: CcdMachineSettingsStore, parent=None):
        super().__init__(parent)
        self.devices = devices
        self.store = store
        self._machine = store.load()
        self._product = CameraRecipeSettings()
        self._recipe_name: str | None = None
        self._recipe_product: CameraRecipeSettings | None = None
        self._product_edited_since_recipe = False
        self._applied: tuple[CameraConnectionSettings, AcquisitionSettings, TriggerSettings] | None = None
        self._last_meter_snapshot = MeterWheelSnapshot()
        self._closed = False
        self._auto_save_requests = AutoSaveRequests()
        self._software_monitor: SoftwareTriggerMonitor | None = None
        self._software_capture_lock = threading.Lock()
        self._software_capture_queued = False
        self._inspection_queue: CameraFrameQueue | None = None
        self._inspection_saves_raw = True
        self._inspection_sequence = 0

        self._converter = PreviewFrameConverter(self.preview_image_ready.emit)
        self._save_queue = self._create_save_queue()
        # Queued even when a driver emits on the GUI thread, so status is read after the device updates it.
        self._frame_arrived.connect(self.refresh_camera_status, Qt.ConnectionType.QueuedConnection)
        # Driver and monitor threads only hand work to the GUI thread, which owns camera and meter-wheel commands.
        queued = Qt.ConnectionType.QueuedConnection
        self._external_trigger_arrived.connect(self._apply_external_trigger_meter_wheel_actions, queued)
        self._software_capture_requested.connect(self._execute_software_trigger_capture, queued)
        self._software_monitor_failed.connect(self._on_software_monitor_failed, queued)
        self._auto_save_rejected.connect(self._on_auto_save_rejected, queued)
        self.devices.camera.set_frame_listener(self._on_device_frame)
        self.devices.camera.set_external_trigger_listener(self._on_external_trigger)

        self._meter_wheel_timer = QTimer(self)
        self._meter_wheel_timer.setInterval(METER_WHEEL_POLL_MS)
        self._meter_wheel_timer.timeout.connect(self.poll_meter_wheel)

    # ------------------------------------------------------------------
    # binding and state
    # ------------------------------------------------------------------
    @property
    def load_error(self) -> str:
        return self.store.last_error

    @property
    def machine_settings(self) -> CcdMachineSettings:
        return self._machine

    def attach(self, screen) -> None:
        screen.camera_connect_requested.connect(self.connect_camera)
        screen.camera_disconnect_requested.connect(self.disconnect_camera)
        screen.preview_start_requested.connect(self.start_preview)
        screen.preview_stop_requested.connect(self.stop_preview)
        screen.capture_requested.connect(self.capture_frame)
        screen.snapshot_requested.connect(self.save_snapshot)
        screen.camera_settings_applied.connect(self.apply_camera_settings)
        screen.save_settings_applied.connect(self.apply_save_settings)
        screen.meter_wheel_connect_requested.connect(self.connect_meter_wheel)
        screen.meter_wheel_disconnect_requested.connect(self.disconnect_meter_wheel)
        screen.encoder_set_requested.connect(self.set_encoder)
        screen.encoder_clear_requested.connect(self.clear_encoder)
        screen.compare_set_requested.connect(self.set_compare)
        screen.compare_clear_requested.connect(self.clear_compare)
        screen.compare_increment_requested.connect(self.apply_compare_increment)
        screen.multiple_rate_changed.connect(self.set_multiple_rate)
        screen.reverse_direction_changed.connect(self.set_reverse_direction)
        screen.cmp_out_width_requested.connect(self.set_cmp_out_width)
        screen.extension_channels_applied.connect(self.apply_extension_channels)

        self.camera_status_changed.connect(screen.set_camera_status)
        self.camera_settings_changed.connect(screen.set_camera_settings)
        self.preview_image_ready.connect(screen.set_preview_image)
        self.save_stats_changed.connect(screen.set_save_stats)
        self.meter_wheel_changed.connect(screen.set_meter_wheel_snapshot)
        self.meter_wheel_settings_changed.connect(screen.set_meter_wheel_settings)
        self.software_trigger_monitor_changed.connect(screen.set_software_trigger_monitor_running)

        screen.set_availability(self.devices.camera.availability(), self.devices.meter_wheel.availability())
        screen.set_camera_settings(self.camera_settings_view())
        screen.set_meter_wheel_settings(self._machine.meter_wheel)
        screen.set_camera_status(self.camera_status())
        screen.set_meter_wheel_snapshot(self._last_meter_snapshot)
        screen.set_save_stats(self._save_queue.stats())

    def availability(self) -> tuple[DeviceAvailability, DeviceAvailability]:
        return self.devices.camera.availability(), self.devices.meter_wheel.availability()

    def camera_status(self) -> CameraStatus:
        return self.devices.camera.status()

    @property
    def product_settings(self) -> CameraRecipeSettings:
        return self._product

    def camera_settings_view(self) -> CcdCameraSettingsView:
        return CcdCameraSettingsView(
            connection=self._machine.connection,
            product=self._product,
            save=self._machine.save,
            pending_hardware_write=self.pending_hardware_write(),
            source_text=self.product_source_text(),
        )

    def product_source_text(self) -> str:
        if self._recipe_name is None:
            return "來源：未載入 Recipe，相機參數只用於本次執行。"
        if self._recipe_product is None and self._product_edited_since_recipe:
            return f"來源：Recipe「{self._recipe_name}」未包含相機設定；CCD 頁的參數尚未儲存到 Recipe。"
        if self._recipe_product is None:
            return f"來源：Recipe「{self._recipe_name}」未包含相機設定，沿用目前參數。"
        if self._recipe_product != self._product:
            return f"來源：Recipe「{self._recipe_name}」，已在 CCD 頁修改，尚未儲存到 Recipe。"
        return f"來源：Recipe「{self._recipe_name}」。"

    def pending_hardware_write(self) -> bool:
        if not self.camera_status().connected or self._applied is None:
            return False
        return self._applied != self._hardware_settings()

    def _hardware_settings(self) -> tuple[CameraConnectionSettings, AcquisitionSettings, TriggerSettings]:
        return self._machine.connection, self._product.acquisition, self._product.trigger

    def hardware_trigger(self) -> TriggerSettings | None:
        """Trigger settings written to the connected camera; automation never follows unapplied edits."""
        applied = self._applied
        return None if applied is None else applied[2]

    @property
    def software_trigger_monitor_running(self) -> bool:
        monitor = self._software_monitor
        return monitor is not None and monitor.is_running

    @property
    def pending_auto_saves(self) -> int:
        return self._auto_save_requests.pending

    def camera_monitor_blocker(self) -> str:
        """Why camera-direct inspection cannot start now, or an empty string when it can."""
        availability = self.devices.camera.availability()
        if not availability.available:
            return f"相機不可用：{availability.reason}"
        hardware = self.hardware_trigger()
        if hardware is None or not self.camera_status().connected:
            return "相機未連線，請先到 CCD 控制連線相機。"
        if hardware.mode == TriggerMode.CONTINUOUS:
            return "相機以連續取像連線；相機直連檢測只檢測觸發影像，請改用外部觸發或軟體觸發並重新連線。"
        return ""

    def attach_inspection_queue(self, queue: CameraFrameQueue, monitor_saves_raw: bool = True) -> None:
        """Hand trigger frames to camera monitoring.

        When the monitor saves every inspected frame itself, the snapshot auto-save skips the frames it
        accepted so an 819 MB frame is not written twice; frames the queue rejects keep auto-save.
        """
        self._inspection_sequence = 0
        self._inspection_saves_raw = bool(monitor_saves_raw)
        self._inspection_queue = queue

    def detach_inspection_queue(self) -> None:
        self._inspection_queue = None

    def raw_frame_saver(self) -> RawFrameSaver:
        """Camera-monitor raw saver in the machine-level save format, written through ``.tmp``."""
        image_format = ImageSaveFormat(self._machine.save.image_format)
        return RawFrameSaver(
            extension=image_format.extension,
            write=lambda frame, path: write_frame_atomic(frame, path, image_format),
        )

    def has_pending_saves(self) -> bool:
        return self._save_queue.stats().pending > 0

    def refresh_camera_status(self) -> None:
        self.camera_status_changed.emit(self.camera_status())

    # ------------------------------------------------------------------
    # camera
    # ------------------------------------------------------------------
    def apply_camera_settings(self, connection: CameraConnectionSettings, product: CameraRecipeSettings) -> None:
        """Operator apply from the CCD screen: machine location is saved, product settings stay in session.

        The window decides whether the product settings also become an unsaved Recipe edit.
        """
        if not self._save_machine(replace(self._machine, connection=connection.normalized())):
            return
        self._product = product.normalized()
        self._product_edited_since_recipe = self._recipe_name is not None
        self.camera_settings_changed.emit(self.camera_settings_view())
        self.product_settings_applied.emit(self._product)

    def set_recipe_camera_settings(self, settings: CameraRecipeSettings | None, recipe_name: str) -> None:
        """Adopt a loaded Recipe; a Recipe without a camera section leaves the current parameters unchanged."""
        self._recipe_name = str(recipe_name)
        self._recipe_product = None if settings is None else settings.normalized()
        self._product_edited_since_recipe = False
        if self._recipe_product is not None:
            self._product = self._recipe_product
        self.camera_settings_changed.emit(self.camera_settings_view())
        if self._recipe_product is not None and self.pending_hardware_write():
            self.notice.emit(
                f"Recipe「{self._recipe_name}」的相機設定與相機目前設定不同，需斷線重連才會寫入相機。", "info"
            )

    def connect_camera(self) -> None:
        connection, acquisition, trigger = self._hardware_settings()
        try:
            status = self.devices.camera.connect(connection, acquisition, trigger)
        except DeviceError as exc:
            self.notice.emit(f"相機連線失敗：{exc}", "error")
        else:
            self._applied = (connection, acquisition, trigger)
            self.notice.emit(f"相機已連線：{status.camera_name or '線掃相機'}", "success")
            self.camera_settings_changed.emit(self.camera_settings_view())
        self.refresh_camera_status()

    def disconnect_camera(self) -> None:
        self.stop_software_trigger_monitor()
        self._run_camera_command(self.devices.camera.disconnect, "相機中斷連線失敗")
        self._applied = None
        self._auto_save_requests.clear()
        self.camera_settings_changed.emit(self.camera_settings_view())

    def start_preview(self) -> None:
        trigger = self.hardware_trigger() or self._product.trigger
        if trigger.mode == TriggerMode.SOFTWARE:
            # Software Trigger does not grab continuously: it monitors the meter wheel and snaps one frame per crossing.
            self.start_software_trigger_monitor()
            return
        self._run_camera_command(self.devices.camera.start_preview, "無法開始預覽")

    def stop_preview(self) -> None:
        self.stop_software_trigger_monitor()
        self._run_camera_command(self.devices.camera.stop_preview, "無法停止取像")

    def capture_frame(self) -> None:
        self._run_camera_command(self.devices.camera.capture_frame, "無法擷取影像")

    def apply_save_settings(self, save: SaveSettings) -> None:
        save = save.normalized()
        previous_workers = self._machine.save.max_concurrent_saves
        if not self._save_machine(replace(self._machine, save=save)):
            return
        if save.max_concurrent_saves != previous_workers and not self.has_pending_saves():
            self._save_queue.close(wait=True)
            self._save_queue = self._create_save_queue()
        self.camera_settings_changed.emit(self.camera_settings_view())
        self.notice.emit("存圖設定已保存。", "success")

    def snapshot_directory(self) -> Path:
        folder = self._machine.save.folder
        return Path(folder) if folder else DEFAULT_SNAPSHOT_DIR

    def save_snapshot(self) -> Path | None:
        frame = self.devices.camera.latest_frame()
        if frame is None:
            self.notice.emit("尚無可保留的影像，請先預覽或擷取。", "warning")
            return None
        path = self._save_queue.submit(frame, self.snapshot_directory(), self._machine.save.image_format)
        if path is None:
            self.notice.emit("存圖佇列已滿，請等待目前存圖完成後再保留影像。", "warning")
        return path

    def _run_camera_command(self, command: Callable[[], None], failure_prefix: str) -> None:
        try:
            command()
        except DeviceError as exc:
            self.notice.emit(f"{failure_prefix}：{exc}", "error")
        self.refresh_camera_status()

    def _on_device_frame(self, frame: np.ndarray) -> None:
        # Driver thread: hand off only; display conversion, saving and status refresh happen elsewhere.
        self._converter.submit(frame)
        saved_by_monitor = self._hand_off_for_inspection(frame) and self._inspection_saves_raw
        if software_frame_requests_auto_save(self.hardware_trigger(), self._product):
            self._auto_save_requests.request()
        # Consume the request either way so each frame uses up exactly one auto-save.
        if self._auto_save_requests.consume() and not saved_by_monitor:
            saved = self._save_queue.submit(frame, self.snapshot_directory(), self._machine.save.image_format)
            if saved is None:
                self._auto_save_rejected.emit()
        self._frame_arrived.emit()

    def _hand_off_for_inspection(self, frame: np.ndarray) -> bool:
        # Driver thread. Only trigger frames are inspected; continuous free-run frames are preview only.
        queue = self._inspection_queue
        hardware = self.hardware_trigger()
        if queue is None or hardware is None or hardware.mode == TriggerMode.CONTINUOUS:
            return False
        self._inspection_sequence += 1
        sequence = self._inspection_sequence
        now = datetime.datetime.now()
        return queue.put(
            CapturedFrame(
                image=frame,
                source_name=f"camera_{now:%Y%m%d_%H%M%S}_{now.microsecond // 1000:03d}_{sequence:06d}",
                received_at=time.perf_counter(),
                metadata={
                    "frame_index": sequence,
                    "captured_at": now.isoformat(timespec="milliseconds"),
                    "trigger_mode": hardware.mode.value,
                    "frame_width": int(frame.shape[1]),
                    "frame_height": int(frame.shape[0]),
                },
            )
        )

    def _on_auto_save_rejected(self) -> None:
        self.notice.emit("自動存圖佇列已滿，這張影像未保存。", "warning")

    # ------------------------------------------------------------------
    # trigger automation
    # ------------------------------------------------------------------
    def _on_external_trigger(self) -> None:
        # Driver thread: the auto-save request must be counted before the frame arrives.
        actions = external_trigger_actions(self.hardware_trigger(), self._product, self._machine.meter_wheel)
        if actions.request_auto_save:
            self._auto_save_requests.request()
        if actions.compare_value is not None:
            self._external_trigger_arrived.emit(actions)

    def _apply_external_trigger_meter_wheel_actions(self, actions: ExternalTriggerActions) -> None:
        if self._closed:
            return
        meter_wheel = self.devices.meter_wheel
        if not meter_wheel.is_connected:
            self.status_message.emit("收到外部觸發，但米輪未連線，未寫入 Compare。")
            return
        try:
            meter_wheel.set_compare(actions.compare_value)
            if actions.encoder_value is not None:
                meter_wheel.set_encoder(actions.encoder_value)
        except DeviceError as exc:
            self.notice.emit(f"外部觸發的米輪動作失敗：{exc}", "error")
            return
        message = f"外部觸發：已寫入 Compare {actions.compare_value}"
        if actions.encoder_value is not None:
            message += f"、Encoder {actions.encoder_value}"
        self.status_message.emit(message + "。")
        self.poll_meter_wheel()

    def start_software_trigger_monitor(self) -> bool:
        if self.software_trigger_monitor_running:
            return True
        hardware = self.hardware_trigger()
        status = self.camera_status()
        reason = ""
        if hardware is None or not status.connected:
            reason = "相機未連線。"
        elif hardware.mode != TriggerMode.SOFTWARE:
            reason = "相機目前不是以軟體觸發連線，請斷線重連以寫入觸發設定。"
        elif status.state != CameraState.IDLE:
            reason = f"相機{CAMERA_STATE_LABELS[status.state]}，請等待完成。"
        elif not self.devices.meter_wheel.is_connected:
            reason = "米輪未連線。"
        if reason:
            self.notice.emit(f"軟體觸發監控未啟動：{reason}", "warning")
            self.refresh_camera_status()
            return False
        monitor = SoftwareTriggerMonitor(
            self.devices.meter_wheel,
            self._machine.meter_wheel.compare_value,
            self._request_software_capture,
            on_message=self.status_message.emit,
            on_error=lambda error: self._software_monitor_failed.emit(str(error)),
            poll_interval_sec=self.software_trigger_poll_sec,
        )
        with self._software_capture_lock:
            self._software_capture_queued = False
        self._software_monitor = monitor
        monitor.start()
        self.software_trigger_monitor_changed.emit(True)
        self.refresh_camera_status()
        return True

    def stop_software_trigger_monitor(self) -> None:
        monitor, self._software_monitor = self._software_monitor, None
        if monitor is None:
            return
        monitor.stop()
        with self._software_capture_lock:
            self._software_capture_queued = False
        self.software_trigger_monitor_changed.emit(False)
        self.status_message.emit("軟體觸發監控已停止；擷取中的影像會繼續收完。")

    def _request_software_capture(self, compare_value: int, encoder_value: int) -> None:
        # Monitor thread: drop requests while one is still waiting for the GUI thread.
        with self._software_capture_lock:
            if self._software_capture_queued:
                return
            self._software_capture_queued = True
        self._software_capture_requested.emit(compare_value, encoder_value)

    def _execute_software_trigger_capture(self, compare_value: int, encoder_value: int) -> None:
        try:
            hardware = self.hardware_trigger()
            if self._closed or self._software_monitor is None or hardware is None or hardware.mode != TriggerMode.SOFTWARE:
                return
            try:
                self.devices.camera.capture_frame()
            except DeviceError as exc:
                self.status_message.emit(
                    f"軟體觸發無法開始擷取（Compare {compare_value}、Encoder {encoder_value}）：{exc}"
                )
            else:
                self.status_message.emit(f"軟體觸發已開始擷取（Compare {compare_value}、Encoder {encoder_value}）。")
        finally:
            with self._software_capture_lock:
                self._software_capture_queued = False
            self.refresh_camera_status()

    def _on_software_monitor_failed(self, message: str) -> None:
        self.stop_software_trigger_monitor()
        self.notice.emit(f"軟體觸發監控失敗，已停止：{message}", "error")

    # ------------------------------------------------------------------
    # meter wheel
    # ------------------------------------------------------------------
    def connect_meter_wheel(self, card_id: int | None = None, quiet: bool = False) -> bool:
        settings = self._machine.meter_wheel
        if card_id is not None and int(card_id) != settings.card_id:
            settings = replace(settings, card_id=int(card_id)).normalized()
            if not self._save_meter_wheel(settings):
                return False
        try:
            self.devices.meter_wheel.connect(settings)
        except DeviceError as exc:
            kind = "warning" if quiet else "error"
            prefix = "米輪自動連線失敗" if quiet else "米輪連線失敗"
            self.notice.emit(f"{prefix}：{exc}", kind)
            self.logger.warning("Meter wheel connect failed: %s", exc)
            return False
        if not quiet:
            self.notice.emit(f"米輪已連線（卡片 ID {settings.card_id}）。", "success")
        self._meter_wheel_timer.start()
        self.poll_meter_wheel()
        return True

    def auto_connect_meter_wheel(self) -> None:
        if self._closed or self.devices.meter_wheel.is_connected:
            return
        if not self.devices.meter_wheel.availability().available:
            return
        self.connect_meter_wheel(quiet=True)

    def disconnect_meter_wheel(self) -> None:
        self.stop_software_trigger_monitor()
        self._meter_wheel_timer.stop()
        self.devices.meter_wheel.disconnect()
        self._publish_meter_snapshot(MeterWheelSnapshot())

    def poll_meter_wheel(self) -> None:
        meter_wheel = self.devices.meter_wheel
        if not meter_wheel.is_connected:
            self._meter_wheel_timer.stop()
            self._publish_meter_snapshot(MeterWheelSnapshot())
            return
        try:
            snapshot = MeterWheelSnapshot(
                connected=True,
                encoder_value=meter_wheel.read_encoder(),
                compare_value=meter_wheel.read_compare(),
                extension_status=tuple(meter_wheel.read_extension_status()),
            )
        except DeviceError as exc:
            self.stop_software_trigger_monitor()
            self._meter_wheel_timer.stop()
            meter_wheel.disconnect()
            self.notice.emit(f"米輪讀值失敗，已中斷連線：{exc}", "error")
            snapshot = MeterWheelSnapshot()
        self._publish_meter_snapshot(snapshot)

    def set_encoder(self, value: int) -> None:
        self._meter_wheel_change({"encoder_value": int(value)}, self.devices.meter_wheel.set_encoder, "encoder_value")

    def set_compare(self, value: int) -> None:
        # The compare value is operator-defined; it is never replaced by the live encoder value.
        self._meter_wheel_change({"compare_value": int(value)}, self.devices.meter_wheel.set_compare, "compare_value")

    def clear_encoder(self) -> None:
        # Clearing writes 0 to the card only; the saved Encoder origin value is kept.
        self._meter_wheel_write(lambda: self.devices.meter_wheel.set_encoder(0))

    def clear_compare(self) -> None:
        self._meter_wheel_write(lambda: self.devices.meter_wheel.set_compare(0))

    def apply_compare_increment(self, value: int) -> None:
        self._meter_wheel_change(
            {"compare_increment": int(value)}, self.devices.meter_wheel.set_compare_increment, "compare_increment"
        )

    def set_multiple_rate(self, rate: MultipleRate) -> None:
        self._meter_wheel_change(
            {"multiple_rate": MultipleRate(rate)}, self.devices.meter_wheel.set_multiple_rate, "multiple_rate"
        )

    def set_reverse_direction(self, reverse: bool) -> None:
        self._meter_wheel_change(
            {"reverse_direction": bool(reverse)}, self.devices.meter_wheel.set_reverse_direction, "reverse_direction"
        )

    def set_cmp_out_width(self, width: int) -> None:
        self._meter_wheel_change({"cmp_out_width": int(width)}, self.devices.meter_wheel.set_cmp_out_width, "cmp_out_width")

    def apply_extension_channels(self, channels: Sequence[ExtensionCompareChannel]) -> None:
        normalized = tuple(channel.normalized() for channel in channels)
        self._meter_wheel_change(
            {"extension_channels": normalized},
            self.devices.meter_wheel.apply_extension_channels,
            "extension_channels",
        )

    def _meter_wheel_change(self, changes: dict, write: Callable[[object], None], field_name: str) -> None:
        settings = replace(self._machine.meter_wheel, **changes).normalized()
        if not self._save_meter_wheel(settings):
            return
        if self.devices.meter_wheel.is_connected:
            self._meter_wheel_write(lambda: write(getattr(settings, field_name)))

    def _meter_wheel_write(self, command: Callable[[], None]) -> None:
        if not self.devices.meter_wheel.is_connected:
            self.notice.emit("米輪未連線。", "warning")
            return
        try:
            command()
        except DeviceError as exc:
            self.notice.emit(f"米輪寫入失敗：{exc}", "error")
            return
        self.poll_meter_wheel()

    def _save_meter_wheel(self, settings: MeterWheelSettings) -> bool:
        if not self._save_machine(replace(self._machine, meter_wheel=settings)):
            return False
        self.meter_wheel_settings_changed.emit(settings)
        return True

    def _publish_meter_snapshot(self, snapshot: MeterWheelSnapshot) -> None:
        if snapshot != self._last_meter_snapshot:
            self._last_meter_snapshot = snapshot
            self.meter_wheel_changed.emit(snapshot)

    # ------------------------------------------------------------------
    # persistence and lifecycle
    # ------------------------------------------------------------------
    def _save_machine(self, settings: CcdMachineSettings) -> bool:
        settings = settings.normalized()
        try:
            self.store.save(settings)
        except OSError as exc:
            self.notice.emit(f"CCD 機台設定檔寫入失敗：{self.store.path}（{exc}）", "error")
            return False
        self._machine = settings
        return True

    def _create_save_queue(self) -> SnapshotSaveQueue:
        return SnapshotSaveQueue(
            max_workers=self._machine.save.max_concurrent_saves,
            listener=self._on_save_stats,
        )

    def _on_save_stats(self, stats: SaveQueueStats) -> None:
        self.save_stats_changed.emit(stats)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.detach_inspection_queue()
        self.stop_software_trigger_monitor()
        self._meter_wheel_timer.stop()
        self.devices.camera.set_frame_listener(None)
        self.devices.camera.set_external_trigger_listener(None)
        self._converter.close()
        self._save_queue.close(wait=True)
        try:
            self.devices.close()
        except DeviceError as exc:
            self.logger.warning("CCD device cleanup failed: %s", exc)
