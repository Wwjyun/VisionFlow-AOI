from __future__ import annotations

import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from core.image_loader import BmpReader, ImageLoadError, ImageLoader


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

    @staticmethod
    def _opencv(path: Path) -> np.ndarray:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)

    @staticmethod
    def _palette_bmp(indices: np.ndarray, palette_bgr: np.ndarray, top_down: bool, colors_used: int) -> bytes:
        """Hand-built 8-bit BI_RGB BMP, since OpenCV only writes grayscale palettes."""
        height, width = indices.shape
        stride = ((width * 8 + 31) // 32) * 4
        entries = len(palette_bgr)
        palette = np.zeros((entries, 4), dtype=np.uint8)
        palette[:, :3] = palette_bgr
        pixel_offset = 14 + 40 + entries * 4
        rows = indices if top_down else indices[::-1]
        pixels = bytearray()
        for row in rows:
            pixels += bytes(row) + b"\0" * (stride - width)
        info = struct.pack(
            "<IiiHHIIiiII", 40, width, -height if top_down else height, 1, 8, 0, len(pixels), 2835, 2835,
            colors_used, 0,
        )
        file_header = struct.pack("<2sIHHI", b"BM", pixel_offset + len(pixels), 0, 0, pixel_offset)
        return file_header + info + palette.tobytes() + bytes(pixels)

    def test_bmp_reader_matches_opencv_for_24_bit_row_padding(self):
        rng = np.random.default_rng(7)
        reader = BmpReader(max_workers=1)
        with tempfile.TemporaryDirectory() as directory:
            for width in range(1, 9):
                for height in (1, 2, 5):
                    path = Path(directory) / f"color_{width}_{height}.bmp"
                    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
                    ok, encoded = cv2.imencode(".bmp", image)
                    self.assertTrue(ok)
                    encoded.tofile(str(path))
                    decoded = reader.read(path)
                    self.assertIsNotNone(decoded, (width, height))
                    np.testing.assert_array_equal(decoded, self._opencv(path))
                    np.testing.assert_array_equal(decoded, image)

    def test_bmp_reader_matches_opencv_for_palettes_and_top_down_rows(self):
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as directory:
            gray_path = Path(directory) / "gray.bmp"
            gray = rng.integers(0, 256, size=(9, 13), dtype=np.uint8)
            ok, encoded = cv2.imencode(".bmp", gray)
            self.assertTrue(ok)
            encoded.tofile(str(gray_path))
            np.testing.assert_array_equal(BmpReader().read(gray_path), self._opencv(gray_path))

            palette = rng.integers(0, 256, size=(6, 3), dtype=np.uint8)
            indices = rng.integers(0, 8, size=(7, 10), dtype=np.uint8)  # 6 and 7 fall outside the palette
            for top_down in (False, True):
                path = Path(directory) / f"palette_{top_down}.bmp"
                path.write_bytes(self._palette_bmp(indices, palette, top_down, colors_used=6))
                decoded = BmpReader().read(path)
                self.assertIsNotNone(decoded)
                np.testing.assert_array_equal(decoded, self._opencv(path))

    def test_parallel_bands_match_the_single_band_read(self):
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "中文大圖.bmp"
            image = rng.integers(0, 256, size=(131, 257, 3), dtype=np.uint8)
            ok, encoded = cv2.imencode(".bmp", image)
            self.assertTrue(ok)
            encoded.tofile(str(path))
            parallel = BmpReader(max_workers=4)
            parallel.PARALLEL_MIN_BYTES = 1
            np.testing.assert_array_equal(parallel.read(path), self._opencv(path))
            np.testing.assert_array_equal(ImageLoader(bmp_reader=parallel).load_bgr(path), image)

    def test_unsupported_or_truncated_bmp_falls_back_to_opencv(self):
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as directory:
            bgra_path = Path(directory) / "bgra.bmp"
            bgra = rng.integers(0, 256, size=(6, 5, 4), dtype=np.uint8)
            ok, encoded = cv2.imencode(".bmp", bgra)
            self.assertTrue(ok)
            encoded.tofile(str(bgra_path))
            self.assertIsNone(BmpReader().read(bgra_path))
            np.testing.assert_array_equal(ImageLoader().load_bgr(bgra_path), self._opencv(bgra_path))

            color_path = Path(directory) / "truncated.bmp"
            ok, encoded = cv2.imencode(".bmp", rng.integers(0, 256, size=(20, 20, 3), dtype=np.uint8))
            self.assertTrue(ok)
            color_path.write_bytes(encoded.tobytes()[:-100])
            self.assertIsNone(BmpReader().read(color_path))

            fake_path = Path(directory) / "not_really.bmp"
            fake_path.write_bytes(b"BM" + b"\0" * 10)
            self.assertIsNone(BmpReader().read(fake_path))
            with self.assertRaises(ImageLoadError):
                ImageLoader().load_bgr(fake_path)

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
