from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QImage

from core.logging_system import LogMixin
from devices.ccd_models import (
    AcquisitionSettings,
    CameraConnectionSettings,
    CameraRecipeSettings,
    CameraStatus,
    CcdMachineSettings,
    DeviceAvailability,
    DeviceError,
    ExtensionCompareChannel,
    MeterWheelSettings,
    MeterWheelSnapshot,
    MultipleRate,
    SaveSettings,
    TriggerSettings,
)
from devices.ccd_settings_store import CcdMachineSettingsStore
from devices.factory import CcdDevices
from devices.frame_writer import SaveQueueStats, SnapshotSaveQueue

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
    notice = Signal(str, str)
    _frame_arrived = Signal()

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

        self._converter = PreviewFrameConverter(self.preview_image_ready.emit)
        self._save_queue = self._create_save_queue()
        # Queued even when a driver emits on the GUI thread, so status is read after the device updates it.
        self._frame_arrived.connect(self.refresh_camera_status, Qt.ConnectionType.QueuedConnection)
        self.devices.camera.set_frame_listener(self._on_device_frame)

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
        self._run_camera_command(self.devices.camera.disconnect, "相機中斷連線失敗")
        self._applied = None
        self.camera_settings_changed.emit(self.camera_settings_view())

    def start_preview(self) -> None:
        self._run_camera_command(self.devices.camera.start_preview, "無法開始預覽")

    def stop_preview(self) -> None:
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
        # Driver thread: hand off only; display conversion and status refresh happen elsewhere.
        self._converter.submit(frame)
        self._frame_arrived.emit()

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
        self._meter_wheel_timer.stop()
        self.devices.camera.set_frame_listener(None)
        self._converter.close()
        self._save_queue.close(wait=True)
        try:
            self.devices.close()
        except DeviceError as exc:
            self.logger.warning("CCD device cleanup failed: %s", exc)
