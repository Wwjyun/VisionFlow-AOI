from __future__ import annotations

import os
import sys
import threading

from pathlib import Path

import numpy as np

from core.logging_system import LogMixin


class HostImageBufferPool(LogMixin):
    """Session-owned host backing reused by consecutive decoded file-order BMP images.

    A fresh 16384x13000 decoded image costs page faults while the reader fills it and a
    ~28 ms free after every inspection. Reusing one backing across the serialized runs of a
    ``GpuExecutionSession`` removes both, and a backing registered as CUDA pinned memory also
    shortens the whole-image H2D (RTX 3090, production shape: read+upload+release 214.5 ms
    fresh, 173.8 ms reused pageable, 155.0 ms reused pinned).

    Only one lease exists at a time; a second concurrent request, a closed pool or ``off`` mode
    returns ``None`` and the reader allocates a fresh array as before. A backing is reused only
    when nothing but the pool still references it: if a view of the previous image survived its
    run, the backing is detached (and unpinned) instead of being overwritten under that view.

    The backing also remembers which file it holds. Inspecting the same unchanged file again (same
    resolved path, size, modification time and file ID) reuses the decoded pixels instead of reading
    them, which is the GUI re-inspection and warm-up-then-inspect case. Pooled images are handed to
    the pipeline read-only, so no step can alter the cached pixels in place.

    ``AOI_HOST_IMAGE_BUFFER`` selects ``auto`` (reuse, pin when the CUDA DLL exports host
    registration), ``pageable`` (reuse, never pin) or ``off`` (always allocate).
    """

    MODES = ("auto", "pageable", "off")
    ENVIRONMENT_VARIABLE = "AOI_HOST_IMAGE_BUFFER"

    def __init__(self, runtime=None, mode: str | None = None):
        requested = str(mode if mode is not None else os.getenv(self.ENVIRONMENT_VARIABLE, "auto"))
        requested = requested.strip().lower() or "auto"
        if requested not in self.MODES:
            self.logger.warning(
                "Unknown %s=%r; using auto", self.ENVIRONMENT_VARIABLE, requested
            )
            requested = "auto"
        self.mode = requested
        self._runtime = runtime
        self._lock = threading.Lock()
        self._buffer: np.ndarray | None = None
        self._baseline_refs = 0
        self._pinned = False
        self._leased = False
        self._closed = False
        self._last_lease: dict = {}
        self._source: tuple | None = None
        self._source_layout: tuple[int, int, bool] | None = None
        self.allocation_count = 0
        self.reuse_count = 0
        self.detached_count = 0
        self.decode_skipped_count = 0

    @property
    def enabled(self) -> bool:
        return self.mode != "off" and not self._closed

    def acquire(self, shape: tuple[int, int]) -> np.ndarray | None:
        """Lease a writable ``uint8`` backing of exactly ``shape``, or ``None`` to allocate normally."""
        shape = (int(shape[0]), int(shape[1]))
        with self._lock:
            if not self.enabled or self._leased or shape[0] <= 0 or shape[1] <= 0:
                return None
            reused = self._buffer is not None and self._buffer.shape == shape
            # The reader is about to overwrite the backing, so it no longer holds any known file.
            self._source = None
            self._source_layout = None
            if reused:
                self.reuse_count += 1
            else:
                self._drop_locked()
                self._buffer = np.empty(shape, dtype=np.uint8)
                self._baseline_refs = self._references_locked()
                self.allocation_count += 1
                if self.mode == "auto":
                    self._pinned = self._pin_locked(self._buffer)
            self._leased = True
            self._last_lease = {
                "reused": reused, "pinned": self._pinned, "nbytes": int(self._buffer.nbytes),
                "decode_skipped": False,
            }
            return self._buffer

    def acquire_cached(self, source: tuple) -> tuple[np.ndarray, tuple[int, int, bool]] | None:
        """Lease the backing without reading when it still holds exactly ``source``."""
        with self._lock:
            if (
                not self.enabled or self._leased or self._buffer is None
                or source is None or self._source != source or self._source_layout is None
            ):
                return None
            self._leased = True
            self.reuse_count += 1
            self.decode_skipped_count += 1
            self._last_lease = {
                "reused": True, "pinned": self._pinned, "nbytes": int(self._buffer.nbytes),
                "decode_skipped": True,
            }
            return self._buffer, self._source_layout

    def record_source(self, source: tuple, image: np.ndarray) -> bool:
        """Remember that the leased backing now holds ``source``; False when ``image`` is not pooled."""
        with self._lock:
            buffer = self._buffer
            if (
                not self._leased or buffer is None
                or image.ndim != 3 or image.shape[2] != 3 or not np.may_share_memory(image, buffer)
            ):
                return False
            if source is not None:
                self._source = source
                self._source_layout = (int(image.shape[0]), int(image.shape[1]), int(image.strides[0]) < 0)
            return True

    def lease(self) -> "HostImageLease":
        """Per-run handle: pass it as the reader's backing provider and close it after the run."""
        return HostImageLease(self)

    def _release_current(self) -> None:
        """Return the leased backing; callers must not hold the array while releasing it."""
        with self._lock:
            if not self._leased:
                return
            self._leased = False
            if self._references_locked() != self._baseline_refs:
                self.logger.warning(
                    "Decoded image pixels are still referenced after inspection; "
                    "allocating a new host image buffer instead of overwriting them"
                )
                self.detached_count += 1
                self._drop_locked()
            elif self._closed:
                self._drop_locked()

    def last_lease(self) -> dict:
        with self._lock:
            return dict(self._last_lease)

    def close(self) -> None:
        """Unpin and forget the backing; a lease still in progress is dropped when released."""
        with self._lock:
            self._closed = True
            if not self._leased:
                self._drop_locked()

    def _references_locked(self) -> int:
        return sys.getrefcount(self._buffer)

    def _pin_locked(self, buffer: np.ndarray) -> bool:
        runtime = self._runtime
        if runtime is None or not bool(getattr(runtime, "supports_host_register", False)):
            return False
        try:
            runtime.register_host_buffer(buffer)
        except Exception as exc:  # pinning is an optimization; pageable reuse stays correct
            self.logger.warning("Host image buffer pinning failed, using pageable memory: %s", exc)
            return False
        return True

    def _drop_locked(self) -> None:
        buffer = self._buffer
        self._buffer = None
        self._baseline_refs = 0
        self._source = None
        self._source_layout = None
        if buffer is not None and self._pinned:
            try:
                self._runtime.unregister_host_buffer(buffer)
            except Exception as exc:
                self.logger.warning("Host image buffer unpinning failed: %s", exc)
        self._pinned = False


class HostImageLease:
    """One run's access to a ``HostImageBufferPool`` backing.

    The lease never stores the array, so after the run drops its image the pool can tell whether
    any pixel view is still alive. Calling the lease is the reader's backing provider; only the
    first successful request in a run receives the pooled backing.
    """

    def __init__(self, pool: HostImageBufferPool):
        self._pool = pool
        self._active = False
        self._pending_source: tuple | None = None
        self.details: dict = {}

    @staticmethod
    def source_identity(path) -> tuple | None:
        """File identity used to reuse decoded pixels: resolved path, size, mtime and file ID."""
        try:
            resolved = Path(path).resolve()
            stat = resolved.stat()
        except OSError:
            return None
        return (str(resolved), int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ino))

    def cached_image(self, path) -> np.ndarray | None:
        """Return the still-decoded pixels of an unchanged ``path`` read-only, or ``None`` to read it."""
        if self._active:
            return None
        self._pending_source = self.source_identity(path)
        hit = self._pool.acquire_cached(self._pending_source)
        if hit is None:
            return None
        self._active = True
        self.details = self._pool.last_lease()
        rows, width, bottom_up = hit[1]
        pixels = hit[0][:, : width * 3].reshape(rows, width, 3)
        image = pixels[::-1] if bottom_up else pixels
        image.flags.writeable = False
        return image

    def adopt(self, path, image: np.ndarray) -> np.ndarray:
        """After a read, make a pooled image read-only and remember its file if it did not change."""
        if not self._active or image is None:
            return image
        source = self._pending_source
        if source is not None and self.source_identity(path) != source:
            source = None  # the file changed while it was read; never reuse these pixels
        if self._pool.record_source(source, image):
            image.flags.writeable = False
        return image

    def __call__(self, shape: tuple[int, int]) -> np.ndarray | None:
        if self._active:
            return None
        buffer = self._pool.acquire(shape)
        if buffer is None:
            return None
        self._active = True
        self.details = self._pool.last_lease()
        return buffer

    def close(self) -> None:
        if self._active:
            self._active = False
            self._pool._release_current()

    def __enter__(self) -> "HostImageLease":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
