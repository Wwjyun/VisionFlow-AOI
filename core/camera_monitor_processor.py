from __future__ import annotations

import datetime
import gc
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.csv_summary import CsvSummaryExporter
from core.gpu_session import GpuExecutionSession
from core.logging_system import LogMixin
from core.monitor_processor import (
    MonitorImageResult,
    MonitorItemCallback,
    MonitorProgressCallback,
    MonitorStopCallback,
)
from core.pipeline import AOIPipeline
from core.result_compactor import compact_inspection_result

# Frames can be hundreds of MB (16384 x 50000 mono is 819 MB), so the hand-off queue stays small and
# a full queue reports the frame as not inspected instead of growing without bound.
CAMERA_FRAME_QUEUE_CAPACITY = 4
QUEUE_POLL_SEC = 0.1


@dataclass(frozen=True)
class CapturedFrame:
    image: np.ndarray
    source_name: str
    received_at: float
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DroppedFrame:
    source_name: str
    received_at: float
    metadata: dict = field(default_factory=dict)


class CameraFrameQueue:
    """Bounded, thread-safe hand-off from a camera driver thread to one inspection worker."""

    def __init__(self, capacity: int = CAMERA_FRAME_QUEUE_CAPACITY):
        self.capacity = max(1, int(capacity))
        self._condition = threading.Condition()
        self._frames: deque[CapturedFrame] = deque()
        self._dropped: deque[DroppedFrame] = deque()
        self._closed = False

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def put(self, frame: CapturedFrame) -> bool:
        with self._condition:
            if self._closed:
                return False
            if len(self._frames) >= self.capacity:
                # Only the name is kept; the pixels are released immediately.
                self._dropped.append(DroppedFrame(frame.source_name, frame.received_at, dict(frame.metadata)))
                self._condition.notify()
                return False
            self._frames.append(frame)
            self._condition.notify()
            return True

    def get(self, timeout: float) -> CapturedFrame | None:
        with self._condition:
            if not self._frames:
                self._condition.wait(timeout)
            return self._frames.popleft() if self._frames else None

    def take_dropped(self) -> list[DroppedFrame]:
        with self._condition:
            dropped = list(self._dropped)
            self._dropped.clear()
            return dropped

    def pending(self) -> int:
        with self._condition:
            return len(self._frames)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class CameraMonitorProcessor(LogMixin):
    """Inspect camera frames in arrival order with one shared GPU session.

    Items use the folder monitor's dictionary shape so the Monitor table, scatter charts and CSV
    summary work unchanged. When stopped, frames that were already queued are still inspected; a
    frame that could not be queued is reported as an ERROR item rather than silently skipped.
    """

    def __init__(
        self,
        frame_queue: CameraFrameQueue,
        recipe_path: Path,
        output_dir: Path,
        output_overrides: dict | None = None,
        progress_callback: MonitorProgressCallback | None = None,
        item_callback: MonitorItemCallback | None = None,
        stop_callback: MonitorStopCallback | None = None,
        warmup_image_path: Path | None = None,
        gpu_session: GpuExecutionSession | None = None,
    ):
        self.frame_queue = frame_queue
        self.recipe_path = Path(recipe_path)
        self.output_dir = Path(output_dir)
        self.output_overrides = output_overrides
        self.progress_callback = progress_callback
        self.item_callback = item_callback
        self.stop_callback = stop_callback
        self.warmup_image_path = Path(warmup_image_path) if warmup_image_path else None
        self.gpu_session = gpu_session
        self._processed_count = 0
        self._dropped_count = 0

    def run(self) -> dict:
        started_at = datetime.datetime.now()
        monitor_output_dir = self.output_dir / "monitor" / f"{started_at:%Y%m%d_%H%M%S}_camera"
        monitor_output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Camera monitor started: recipe=%s output=%s", self.recipe_path, monitor_output_dir)
        session_started = time.perf_counter()
        with GpuExecutionSession.scoped(self.recipe_path, self.gpu_session) as gpu_session:
            session_ms = round((time.perf_counter() - session_started) * 1000.0, 1)
            gpu_warmup = {
                "session_ms": session_ms,
                **gpu_session.warm_up_before_run(
                    self.recipe_path,
                    self.warmup_image_path,
                    progress_callback=lambda pct, msg: self._progress(pct, msg),
                ),
            }
            self.logger.info("Camera monitor GPU warm-up: %s", gpu_warmup)
            self._progress(0, f"{GpuExecutionSession.warm_up_notice(gpu_warmup)}等待相機觸發影像")
            while not self._should_stop():
                self._report_dropped()
                frame = self.frame_queue.get(QUEUE_POLL_SEC)
                if frame is not None:
                    self._inspect(frame, monitor_output_dir, gpu_session)
            self.frame_queue.close()
            remaining = self.frame_queue.pending()
            if remaining:
                self._progress(0, f"監控停止中，完成佇列中的 {remaining} 張影像")
            while (frame := self.frame_queue.get(0)) is not None:
                self._inspect(frame, monitor_output_dir, gpu_session)
            self._report_dropped()

        finished_at = datetime.datetime.now()
        summary = {
            "source": "camera",
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "duration_sec": round((finished_at - started_at).total_seconds(), 2),
            "output_dir": str(monitor_output_dir),
            "processed": self._processed_count,
            "dropped": self._dropped_count,
            "gpu_warmup": gpu_warmup,
        }
        csv_summary_path = CsvSummaryExporter.write_summary(monitor_output_dir / "csv")
        if csv_summary_path is not None:
            summary["csv_summary"] = str(csv_summary_path)
        self.logger.info("Camera monitor stopped: summary=%s", summary)
        return summary

    def _inspect(self, frame: CapturedFrame, monitor_output_dir: Path, gpu_session: GpuExecutionSession) -> None:
        processing_started = time.perf_counter()
        pipeline_duration = 0.0
        result = None
        try:
            self.logger.info("Camera frame inspection started: frame=%s shape=%s", frame.source_name, frame.image.shape)
            pipeline = AOIPipeline(
                recipe_path=self.recipe_path,
                output_dir=monitor_output_dir,
                output_overrides=self.output_overrides,
                gpu_session=gpu_session,
                progress_callback=lambda pct, msg: self._progress(pct, f"{frame.source_name}: {msg}"),
            )
            result = pipeline.run_frame(frame.image, frame.source_name, frame.metadata)
            pipeline_duration = float(result.get("duration_sec", 0) or 0)
            summary = result.get("summary", {})
            item = MonitorImageResult(
                image_path=Path(frame.source_name),
                final_result=str(result.get("final_result", "-")),
                defect_count=int(summary.get("defect_count", 0)),
                ng_count=int(summary.get("ng_count", 0)),
                tile_count=int(summary.get("tile_count", 0)),
                duration_sec=0.0,
                outputs=result.get("outputs", {}),
                detail=compact_inspection_result(result),
                timing={},
            )
        except Exception as exc:
            self.logger.exception("Camera frame inspection failed: frame=%s", frame.source_name)
            item = self._error_item(frame.source_name, str(exc))
        finally:
            result = None
            gc.collect(0)
        self._emit(item, frame.received_at, processing_started, pipeline_duration, frame.metadata)
        self._processed_count += 1
        self._progress(100, f"已處理 {frame.source_name}")

    def _report_dropped(self) -> None:
        for dropped in self.frame_queue.take_dropped():
            self._dropped_count += 1
            self.logger.warning("Camera frame dropped: frame=%s capacity=%s", dropped.source_name, self.frame_queue.capacity)
            item = self._error_item(
                dropped.source_name,
                f"檢測佇列已滿（上限 {self.frame_queue.capacity} 張），此影像未檢測。",
            )
            now = time.perf_counter()
            self._emit(item, dropped.received_at, now, 0.0, dropped.metadata)

    @staticmethod
    def _error_item(source_name: str, message: str) -> MonitorImageResult:
        return MonitorImageResult(
            image_path=Path(source_name),
            final_result="ERROR",
            defect_count=0,
            ng_count=0,
            tile_count=0,
            duration_sec=0.0,
            outputs={},
            detail={},
            timing={},
            error=message,
        )

    def _emit(
        self,
        item: MonitorImageResult,
        received_at: float,
        processing_started: float,
        pipeline_duration: float,
        metadata: dict,
    ) -> None:
        finished = time.perf_counter()
        # Camera frames skip file discovery and moving: end-to-end runs from the driver hand-off.
        timing = {
            "queue_wait_sec": round(max(0.0, processing_started - received_at), 6),
            "pipeline_and_reports_sec": round(max(0.0, pipeline_duration), 6),
            "end_to_end_sec": round(max(0.0, finished - received_at), 3),
        }
        data = item.to_dict()
        data["duration_sec"] = timing["end_to_end_sec"]
        data["timing"] = timing
        data["source"] = "camera"
        data["camera"] = dict(metadata)
        if self.item_callback is not None:
            self.item_callback(data)

    def _should_stop(self) -> bool:
        return bool(self.stop_callback and self.stop_callback())

    def _progress(self, percent: int, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(max(0, min(100, int(percent))), message)
