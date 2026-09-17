from __future__ import annotations

import datetime
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from devices.ccd_models import ImageSaveFormat

TIFF_COMPRESSION_NONE = 1
TIFF_COMPRESSION_LZW = 5


def encode_frame(frame: np.ndarray, image_format: ImageSaveFormat) -> bytes:
    image_format = ImageSaveFormat(image_format)
    params: list[int] = []
    if image_format == ImageSaveFormat.TIF:
        params = [cv2.IMWRITE_TIFF_COMPRESSION, TIFF_COMPRESSION_LZW]
    elif image_format == ImageSaveFormat.TIF_UNCOMPRESSED:
        params = [cv2.IMWRITE_TIFF_COMPRESSION, TIFF_COMPRESSION_NONE]
    ok, encoded = cv2.imencode(image_format.extension, frame, params)
    if not ok:
        raise RuntimeError(f"影像編碼失敗：{image_format.value}")
    return encoded.tobytes()


def write_frame_atomic(frame: np.ndarray, path: Path, image_format: ImageSaveFormat) -> Path:
    """Write through a `.tmp` file so viewers and folder monitors never see a partial image."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(encode_frame(frame, image_format))
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def unique_snapshot_path(
    directory: Path,
    image_format: ImageSaveFormat,
    now: datetime.datetime | None = None,
    reserved: set[Path] | frozenset[Path] = frozenset(),
) -> Path:
    now = now or datetime.datetime.now()
    stem = f"ccd_{now:%Y%m%d_%H%M%S}_{now.microsecond // 1000:03d}"
    extension = ImageSaveFormat(image_format).extension
    candidate = Path(directory) / f"{stem}{extension}"
    index = 1
    while candidate in reserved or candidate.exists() or candidate.with_name(candidate.name + ".tmp").exists():
        candidate = Path(directory) / f"{stem}_{index}{extension}"
        index += 1
    return candidate


@dataclass(frozen=True)
class SaveQueueStats:
    active: int = 0
    waiting: int = 0
    done: int = 0
    failed: int = 0
    last_path: str = ""
    last_error: str = ""

    @property
    def pending(self) -> int:
        return self.active + self.waiting


class SnapshotSaveQueue:
    """Bounded background writer for full-resolution frames."""

    def __init__(
        self,
        max_workers: int = 2,
        max_pending: int = 16,
        listener: Callable[[SaveQueueStats], None] | None = None,
    ):
        self._max_workers = max(1, int(max_workers))
        self._max_pending = max(1, int(max_pending))
        self._listener = listener
        self._lock = threading.Lock()
        self._stats = SaveQueueStats()
        self._executor: ThreadPoolExecutor | None = None
        self._reserved_paths: set[Path] = set()

    def stats(self) -> SaveQueueStats:
        with self._lock:
            return self._stats

    def submit(self, frame: np.ndarray, directory: Path, image_format: ImageSaveFormat) -> Path | None:
        with self._lock:
            if self._stats.pending >= self._max_pending:
                return None
            path = unique_snapshot_path(Path(directory), image_format, reserved=self._reserved_paths)
            self._reserved_paths.add(path)
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="ccd-save")
            self._stats = replace(self._stats, waiting=self._stats.waiting + 1)
            executor = self._executor
            stats = self._stats
        self._notify(stats)
        executor.submit(self._write, frame, path, ImageSaveFormat(image_format))
        return path

    def close(self, wait: bool = True) -> None:
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=wait)

    def _write(self, frame: np.ndarray, path: Path, image_format: ImageSaveFormat) -> None:
        with self._lock:
            self._stats = replace(self._stats, waiting=self._stats.waiting - 1, active=self._stats.active + 1)
            stats = self._stats
        self._notify(stats)
        error = ""
        try:
            write_frame_atomic(frame, path, image_format)
        except Exception as exc:  # reported to the operator through stats
            error = f"{path.name}：{exc}"
        with self._lock:
            self._reserved_paths.discard(path)
            if error:
                self._stats = replace(
                    self._stats, active=self._stats.active - 1, failed=self._stats.failed + 1, last_error=error
                )
            else:
                self._stats = replace(
                    self._stats, active=self._stats.active - 1, done=self._stats.done + 1, last_path=str(path)
                )
            stats = self._stats
        self._notify(stats)

    def _notify(self, stats: SaveQueueStats) -> None:
        if self._listener is not None:
            self._listener(stats)
