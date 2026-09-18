from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtGui import QImage

# Display-only level-of-detail pyramid for the inspection viewer. Levels never feed
# inspection: overlay, cursor and defect coordinates stay in original image pixels.

# Images whose long side exceeds this get a pyramid; smaller images display directly.
PREVIEW_LOD_MIN_SIDE = 4096
# Halving stops once the long side fits within this overview size.
PREVIEW_OVERVIEW_MAX_SIDE = 2048


@dataclass(frozen=True)
class PreviewImage:
    """A full-resolution RGB preview plus its display pyramid.

    ``levels[0]`` is ``image``; each following level halves the previous one (rounding up)
    with ``INTER_AREA``. A single level means the viewer shows the image directly.
    """

    image: QImage
    levels: tuple[QImage, ...]

    @property
    def width(self) -> int:
        return self.image.width()

    @property
    def height(self) -> int:
        return self.image.height()


def rgb888_view(image: QImage) -> np.ndarray:
    """Return a writable ``(h, w, 3)`` view of an RGB888 QImage buffer (rows may be padded)."""
    if image.format() != QImage.Format.Format_RGB888:
        raise ValueError("rgb888_view requires a Format_RGB888 QImage")
    height, width = image.height(), image.width()
    buffer = np.frombuffer(image.bits(), dtype=np.uint8)
    return np.lib.stride_tricks.as_strided(
        buffer, shape=(height, width, 3), strides=(image.bytesPerLine(), 3, 1), writeable=True
    )


def rgb_qimage_from_bgr(bgr: np.ndarray) -> QImage:
    """Convert a BGR uint8 image straight into a QImage-owned RGB888 buffer."""
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError(f"Expected an 8-bit 3-channel BGR image, got shape={bgr.shape} dtype={bgr.dtype}")
    height, width = bgr.shape[:2]
    image = QImage(width, height, QImage.Format.Format_RGB888)
    if image.isNull():
        raise MemoryError(f"Could not allocate a {width}x{height} preview image")
    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB, dst=rgb888_view(image))
    return image


def build_preview_levels(
    image: QImage,
    lod_min_side: int = PREVIEW_LOD_MIN_SIDE,
    overview_max_side: int = PREVIEW_OVERVIEW_MAX_SIDE,
) -> tuple[QImage, ...]:
    """Build the display pyramid; returns ``(image,)`` when the image is small enough."""
    if image.isNull() or max(image.width(), image.height()) <= lod_min_side:
        return (image,)
    source = image if image.format() == QImage.Format.Format_RGB888 else image.convertToFormat(
        QImage.Format.Format_RGB888
    )
    levels = [image]
    current = source
    while max(current.width(), current.height()) > overview_max_side:
        width = max(1, (current.width() + 1) // 2)
        height = max(1, (current.height() + 1) // 2)
        level = QImage(width, height, QImage.Format.Format_RGB888)
        if level.isNull():
            raise MemoryError(f"Could not allocate a {width}x{height} preview level")
        cv2.resize(rgb888_view(current), (width, height), dst=rgb888_view(level), interpolation=cv2.INTER_AREA)
        levels.append(level)
        current = level
    return tuple(levels)


def preview_image(
    image: QImage,
    lod_min_side: int = PREVIEW_LOD_MIN_SIDE,
    overview_max_side: int = PREVIEW_OVERVIEW_MAX_SIDE,
) -> PreviewImage:
    return PreviewImage(image, build_preview_levels(image, lod_min_side, overview_max_side))
