from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.logging_system import LogMixin


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class ImageLoadError(RuntimeError):
    pass


class ImageLoader(LogMixin):
    def __init__(self, supported_extensions: set[str] | None = None):
        self.supported_extensions = supported_extensions or SUPPORTED_EXTENSIONS

    def load_bgr(self, path: Path):
        image_path = self._validate_path(path)
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


def load_image(path: Path):
    return ImageLoader().load_bgr(path)
