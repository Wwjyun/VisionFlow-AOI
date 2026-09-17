from __future__ import annotations

import os
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.logging_system import LogMixin


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class ImageLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class _BmpLayout:
    pixel_offset: int
    width: int
    height: int
    bits_per_pixel: int
    stride: int
    bottom_up: bool
    palette: np.ndarray | None


class BmpReader:
    """Read uncompressed 24-bit and 8-bit palette BMP files with OpenCV-identical pixels.

    ``cv2.imdecode`` streams a BMP row by row: a 16384x13000 24-bit production image measured 848 ms
    against 241 ms here on 8 threads. Rows are read in parallel bands with positional reads and written
    straight into their flipped destination, so the host keeps one decoded copy plus at most the
    in-flight bands. Anything this reader does not recognise - compression, bit fields, 1/4/16/32
    bits per pixel, truncated or inconsistent headers - returns ``None`` and the caller uses
    ``cv2.imdecode``, which remains the reference for every other layout.
    """

    PARALLEL_MIN_BYTES = 32 * 1024 * 1024
    MAX_WORKERS = 8
    BANDS_PER_WORKER = 2

    def __init__(self, max_workers: int | None = None):
        self.max_workers = max(1, int(max_workers or min(self.MAX_WORKERS, os.cpu_count() or 1)))

    def read(self, path: Path, *, preserve_file_order: bool = False) -> np.ndarray | None:
        path = Path(path)
        with open(path, "rb") as handle:
            header = handle.read(1024)
            layout = self._layout(header, path.stat().st_size, handle)
        if layout is None:
            return None
        rows = abs(layout.height)
        pixel_bytes = layout.stride * rows
        direct_file_rows = bool(preserve_file_order and layout.bits_per_pixel == 24)
        if direct_file_rows:
            backing = np.empty((rows, layout.stride), dtype=np.uint8)
            pixels = backing[:, : layout.width * 3].reshape(rows, layout.width, 3)
            output = pixels[::-1] if layout.bottom_up else pixels
        else:
            backing = None
            output = np.empty((rows, layout.width, 3), dtype=np.uint8)
        workers = self.max_workers if pixel_bytes >= self.PARALLEL_MIN_BYTES else 1
        bands = min(rows, workers * self.BANDS_PER_WORKER)
        bounds = np.linspace(0, rows, bands + 1, dtype=np.int64)
        if bands == 1:
            if direct_file_rows:
                self._read_raw_band(path, layout, backing, 0, rows)
            else:
                self._read_band(path, layout, output, 0, rows)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(
                        self._read_raw_band if direct_file_rows else self._read_band,
                        path,
                        layout,
                        backing if direct_file_rows else output,
                        int(bounds[i]),
                        int(bounds[i + 1]),
                    )
                    for i in range(bands)
                ]
                for future in futures:
                    future.result()
        return output

    @staticmethod
    def _read_raw_band(
        path: Path, layout: _BmpLayout, backing: np.ndarray, start: int, stop: int,
    ) -> None:
        """Read packed file rows directly; the returned image view supplies the row orientation."""
        with open(path, "rb", buffering=0) as handle:
            handle.seek(layout.pixel_offset + start * layout.stride)
            view = memoryview(backing[start:stop]).cast("B")
            done = 0
            while done < len(view):
                received = handle.readinto(view[done:])
                if not received:
                    raise ImageLoadError(f"BMP pixel data ended early: {path}")
                done += received

    @staticmethod
    def _layout(header: bytes, file_size: int, handle) -> _BmpLayout | None:
        if len(header) < 54 or header[:2] != b"BM":
            return None
        pixel_offset = struct.unpack_from("<I", header, 10)[0]
        info_size = struct.unpack_from("<I", header, 14)[0]
        if info_size < 40 or 14 + info_size > pixel_offset:
            return None
        width, height = struct.unpack_from("<ii", header, 18)
        planes, bits_per_pixel = struct.unpack_from("<HH", header, 26)
        compression = struct.unpack_from("<I", header, 30)[0]
        colors_used = struct.unpack_from("<I", header, 46)[0]
        if planes != 1 or compression != 0 or width <= 0 or height == 0 or bits_per_pixel not in (8, 24):
            return None
        rows = abs(height)
        stride = ((width * bits_per_pixel + 31) // 32) * 4
        if pixel_offset + stride * rows > file_size:
            return None
        palette = None
        if bits_per_pixel == 8:
            entries = colors_used or 256
            palette_start = 14 + info_size
            if entries > 256 or palette_start + entries * 4 > pixel_offset:
                return None
            handle.seek(palette_start)
            raw = handle.read(entries * 4)
            if len(raw) != entries * 4:
                return None
            # Indices beyond the stored entries read black, as OpenCV's zero-initialised palette does.
            palette = np.zeros((256, 3), dtype=np.uint8)
            palette[:entries] = np.frombuffer(raw, dtype=np.uint8).reshape(entries, 4)[:, :3]
        return _BmpLayout(pixel_offset, width, height, bits_per_pixel, stride, height > 0, palette)

    @staticmethod
    def _read_band(path: Path, layout: _BmpLayout, output: np.ndarray, start: int, stop: int) -> None:
        """Read file rows [start, stop) and place them at their image rows."""
        count = stop - start
        band = np.empty(count * layout.stride, dtype=np.uint8)
        with open(path, "rb", buffering=0) as handle:
            handle.seek(layout.pixel_offset + start * layout.stride)
            view = memoryview(band)
            done = 0
            while done < band.size:
                received = handle.readinto(view[done:])
                if not received:
                    raise ImageLoadError(f"BMP pixel data ended early: {path}")
                done += received
        rows = band.reshape(count, layout.stride)
        if layout.bits_per_pixel == 24:
            pixels = rows[:, : layout.width * 3].reshape(count, layout.width, 3)
        else:
            pixels = layout.palette[rows[:, : layout.width]]
        total = abs(layout.height)
        if layout.bottom_up:
            output[total - stop : total - start] = pixels[::-1]
        else:
            output[start:stop] = pixels


class ImageLoader(LogMixin):
    def __init__(self, supported_extensions: set[str] | None = None, bmp_reader: BmpReader | None = None):
        self.supported_extensions = supported_extensions or SUPPORTED_EXTENSIONS
        self.bmp_reader = bmp_reader or BmpReader()

    def load_bgr(self, path: Path, *, preserve_bmp_file_order: bool = False):
        image_path = self._validate_path(path)
        if image_path.suffix.lower() == ".bmp":
            try:
                image = self.bmp_reader.read(
                    image_path, preserve_file_order=preserve_bmp_file_order
                )
            except (OSError, ValueError, ImageLoadError):
                self.logger.debug("Fast BMP read failed, using OpenCV: %s", image_path, exc_info=True)
                image = None
            if image is not None:
                self.logger.debug("Image read completed (BMP reader): %s shape=%s", image_path, image.shape)
                return image
        try:
            self.logger.debug("Reading image with OpenCV: %s", image_path)
            # np.fromfile supports Unicode paths on Windows; cv2.imread does not
            # reliably do so. IMREAD_COLOR applies EXIF orientation by default.
            data = np.fromfile(str(image_path), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
            if image is None:
                raise ImageLoadError(f"OpenCV could not decode image: {image_path}")
            self.logger.debug("Image read completed: %s shape=%s", image_path, image.shape)
            return image
        except (OSError, ValueError, cv2.error) as exc:
            self.logger.exception("Image read failed: %s", image_path)
            raise ImageLoadError(f"OpenCV failed to read image: {image_path}") from exc

    def load_rgb(self, path: Path):
        return cv2.cvtColor(self.load_bgr(path), cv2.COLOR_BGR2RGB)

    def _validate_path(self, path: Path) -> Path:
        image_path = Path(path)
        if image_path.suffix.lower() not in self.supported_extensions:
            raise ImageLoadError(f"Unsupported image extension: {image_path.suffix}")
        if not image_path.exists():
            raise ImageLoadError(f"Image does not exist: {image_path}")
        return image_path


def frame_to_bgr(frame: np.ndarray) -> np.ndarray:
    """Convert an acquired camera frame to the BGR ``uint8`` image the pipeline inspects.

    A grayscale frame becomes three equal channels, which is pixel-identical to decoding the same
    frame after saving it as an 8-bit BMP. The returned array is always a new, writable copy, so the
    camera's read-only frame buffer is never shared with inspection.
    """
    image = np.asarray(frame)
    if image.dtype != np.uint8:
        raise ImageLoadError(f"Camera frame must be uint8, got {image.dtype}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return np.array(image, dtype=np.uint8, copy=True, order="C")
    raise ImageLoadError(f"Camera frame must be HxW or HxWx3, got shape {image.shape}")


def load_image(path: Path, *, preserve_bmp_file_order: bool = False):
    return ImageLoader().load_bgr(
        path, preserve_bmp_file_order=preserve_bmp_file_order
    )
