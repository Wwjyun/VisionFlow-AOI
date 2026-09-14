from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from core.image_loader import ImageLoadError, ImageLoader


class ImageLoaderTests(unittest.TestCase):
    def test_unicode_path_color_channels_and_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "中文影像.png"
            bgra = np.array([[[10, 20, 30, 255], [40, 50, 60, 0]]], dtype=np.uint8)
            ok, encoded = cv2.imencode(".png", bgra)
            self.assertTrue(ok)
            encoded.tofile(str(path))

            loader = ImageLoader()
            expected_bgr = bgra[:, :, :3]
            np.testing.assert_array_equal(loader.load_bgr(path), expected_bgr)
            np.testing.assert_array_equal(
                loader.load_rgb(path), cv2.cvtColor(expected_bgr, cv2.COLOR_BGR2RGB)
            )

    def test_exif_orientation_is_applied(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rotated.jpg"
            rgb = np.zeros((12, 18, 3), dtype=np.uint8)
            rgb[:6, :9] = (255, 0, 0)
            rgb[6:, 9:] = (0, 255, 0)
            image = Image.fromarray(rgb)
            exif = Image.Exif()
            exif[274] = 6  # Rotate 90 degrees clockwise.
            image.save(path, exif=exif, quality=100, subsampling=0)

            decoded = ImageLoader().load_rgb(path)
            raw = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            expected = cv2.cvtColor(cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE), cv2.COLOR_BGR2RGB)
            np.testing.assert_array_equal(decoded, expected)

    def test_invalid_image_reports_load_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.png"
            path.write_bytes(b"not an image")
            with self.assertRaises(ImageLoadError):
                ImageLoader().load_bgr(path)

    def test_opencv_limit_is_raised_before_native_import(self):
        script = (
            "import core, cv2, numpy as np; "
            "image=np.zeros((20,20,3),np.uint8); "
            "ok,data=cv2.imencode('.png',image); "
            "assert ok and cv2.imdecode(data,cv2.IMREAD_COLOR).shape==(20,20,3)"
        )
        env = os.environ.copy()
        env["OPENCV_IO_MAX_IMAGE_PIXELS"] = "100"
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
