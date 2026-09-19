from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

from devices.ccd_models import CameraRecipeSettings, DeviceError, MeterWheelSettings, TriggerMode, TriggerSettings
from devices.interfaces import MeterWheel

# ============================================================
# Trigger automation ported from xx_ccd MainForm:
#   ApplyMeterWheelActionsOnExternalTrigger, Queue*AutoSave and
#   RunSoftwareTriggerMeterWheelMonitor.
# Hardware-trigger gating uses the trigger settings written to the camera at connect.
# ============================================================

SOFTWARE_TRIGGER_POLL_SEC = 0.05


@dataclass(frozen=True)
class ExternalTriggerActions:
    compare_value: int | None = None
    encoder_value: int | None = None
    request_auto_save: bool = False


def external_trigger_actions(
    hardware_trigger: TriggerSettings | None,
    product: CameraRecipeSettings,
    meter_wheel: MeterWheelSettings,
) -> ExternalTriggerActions:
    """What one Sapera external-trigger event should do."""
    if hardware_trigger is None or hardware_trigger.mode != TriggerMode.EXTERNAL:
        return ExternalTriggerActions()
    if not hardware_trigger.external_frame_one_frame:
        return ExternalTriggerActions()
    compare_value = meter_wheel.compare_value if hardware_trigger.compare_follows_encoder else None
    encoder_value = (
        meter_wheel.encoder_value
        if hardware_trigger.compare_follows_encoder and hardware_trigger.set_encoder_on_trigger
        else None
    )
    return ExternalTriggerActions(compare_value, encoder_value, product.auto_save_external_one_frame)


def software_frame_requests_auto_save(hardware_trigger: TriggerSettings | None, product: CameraRecipeSettings) -> bool:
    return (
        hardware_trigger is not None
        and hardware_trigger.mode == TriggerMode.SOFTWARE
        and product.auto_save_software_trigger
    )


class AutoSaveRequests:
    """Thread-safe count of frames that should be saved; each completed frame consumes one."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending = 0

    @property
    def pending(self) -> int:
        with self._lock:
            return self._pending

    def request(self) -> None:
        with self._lock:
            self._pending += 1

    def consume(self) -> bool:
        with self._lock:
            if self._pending <= 0:
                self._pending = 0
                return False
            self._pending -= 1
            return True

    def clear(self) -> None:
        with self._lock:
            self._pending = 0


class SoftwareTriggerMonitor:
    """Start one frame each time the encoder moves below the saved compare value.

    After a capture is requested the monitor waits for the encoder to move above the compare
    value before it can arm again, because the card does not keep pulsing while the encoder
    stays below it. Each line of the frame is still triggered by meter-wheel pulses.
    """

    def __init__(
        self,
        meter_wheel: MeterWheel,
        compare_value: int,
        request_capture: Callable[[int, int], None],
        on_message: Callable[[str], None] = lambda _message: None,
        on_error: Callable[[DeviceError], None] = lambda _error: None,
        poll_interval_sec: float = SOFTWARE_TRIGGER_POLL_SEC,
    ):
        self.meter_wheel = meter_wheel
        self.compare_value = int(compare_value)
        self._request_capture = request_capture
        self._on_message = on_message
        self._on_error = on_error
        self._poll_interval_sec = max(0.001, float(poll_interval_sec))
        self._waiting_for_below_compare = True
        self._first_step = True
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def waiting_for_below_compare(self) -> bool:
        return self._waiting_for_below_compare

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    def step(self) -> None:
        """Read the encoder once and act; raises `DeviceError` on meter-wheel failure."""
        compare_value = self.compare_value
        encoder_value = self.meter_wheel.read_encoder()
        first_step, self._first_step = self._first_step, False
        if self._waiting_for_below_compare:
            if encoder_value < compare_value:
                self.meter_wheel.set_compare(compare_value)
                self._waiting_for_below_compare = False
                self._request_capture(compare_value, encoder_value)
                self._on_message(f"軟體觸發：已寫入 Compare {compare_value}（Encoder {encoder_value}），要求擷取一張。")
            elif first_step:
                self._on_message(f"軟體觸發監控中：等待 Encoder 低於 Compare {compare_value}（目前 {encoder_value}）。")
            return
        if encoder_value > compare_value:
            self._waiting_for_below_compare = True
            self._on_message(f"軟體觸發監控中：等待 Encoder 低於 Compare {compare_value}（目前 {encoder_value}）。")

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, name="ccd-software-trigger", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def _run(self) -> None:
        try:
            self.step()
            while not self._stop_event.wait(self._poll_interval_sec):
                self.step()
        except DeviceError as exc:
            self._stop_event.set()
            self._on_error(exc)
