from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from devices.ccd_models import (
    AcquisitionSettings,
    CameraConnectionSettings,
    CameraStatus,
    DeviceAvailability,
    DeviceError,
    ExtensionCompareChannel,
    MeterWheelSettings,
    MultipleRate,
    TriggerSettings,
)
from devices.interfaces import FrameListener, LineScanCamera, MeterWheel
from devices.lsi8181 import Lsi8181Library, Lsi8181MeterWheel
from devices.simulated import SimulatedLineScanCamera, SimulatedMeterWheel

SIMULATOR_ENV = "VISIONFLOW_CCD_SIMULATOR"

CAMERA_BINDING_PENDING_REASON = (
    "Sapera LT 相機綁定尚未實作（P11 pythonnet spike 待相機機台執行）；"
    f"可設定環境變數 {SIMULATOR_ENV}=1 使用模擬相機。"
)


class UnavailableLineScanCamera(LineScanCamera):
    """Placeholder that keeps the application usable when no camera backend exists."""

    def __init__(self, reason: str):
        self._reason = reason

    def availability(self) -> DeviceAvailability:
        return DeviceAvailability(False, self._reason)

    def status(self) -> CameraStatus:
        return CameraStatus(message=self._reason)

    def set_frame_listener(self, listener: FrameListener | None) -> None:
        return None

    def connect(
        self,
        connection: CameraConnectionSettings,
        acquisition: AcquisitionSettings,
        trigger: TriggerSettings,
    ) -> CameraStatus:
        raise DeviceError(self._reason)

    def disconnect(self) -> None:
        return None

    def start_preview(self) -> None:
        raise DeviceError(self._reason)

    def stop_preview(self) -> None:
        return None

    def capture_frame(self) -> None:
        raise DeviceError(self._reason)

    def latest_frame(self) -> np.ndarray | None:
        return None

    def close(self) -> None:
        return None


class UnavailableMeterWheel(MeterWheel):
    def __init__(self, reason: str):
        self._reason = reason

    def availability(self) -> DeviceAvailability:
        return DeviceAvailability(False, self._reason)

    @property
    def is_connected(self) -> bool:
        return False

    def connect(self, settings: MeterWheelSettings) -> None:
        raise DeviceError(self._reason)

    def disconnect(self) -> None:
        return None

    def read_encoder(self) -> int:
        raise DeviceError(self._reason)

    def set_encoder(self, value: int) -> None:
        raise DeviceError(self._reason)

    def read_compare(self) -> int:
        raise DeviceError(self._reason)

    def set_compare(self, value: int) -> None:
        raise DeviceError(self._reason)

    def set_compare_increment(self, value: int) -> None:
        raise DeviceError(self._reason)

    def set_multiple_rate(self, rate: MultipleRate) -> None:
        raise DeviceError(self._reason)

    def set_reverse_direction(self, reverse: bool) -> None:
        raise DeviceError(self._reason)

    def set_cmp_out_width(self, width: int) -> None:
        raise DeviceError(self._reason)

    def read_extension_status(self) -> tuple[bool, ...]:
        raise DeviceError(self._reason)

    def apply_extension_channels(self, channels: Sequence[ExtensionCompareChannel]) -> None:
        raise DeviceError(self._reason)

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class CcdDevices:
    camera: LineScanCamera
    meter_wheel: MeterWheel

    def close(self) -> None:
        try:
            self.camera.close()
        finally:
            self.meter_wheel.close()


def create_ccd_devices(environ: Mapping[str, str] | None = None) -> CcdDevices:
    env = os.environ if environ is None else environ
    if str(env.get(SIMULATOR_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}:
        return CcdDevices(SimulatedLineScanCamera(), SimulatedMeterWheel(auto_advance_per_read=25))
    # The LSI-8181 DLL is loaded lazily; a missing driver only makes the meter wheel unavailable.
    return CcdDevices(
        UnavailableLineScanCamera(CAMERA_BINDING_PENDING_REASON),
        Lsi8181MeterWheel(loader=lambda: Lsi8181Library.load(environ=env)),
    )
