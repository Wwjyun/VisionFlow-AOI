from __future__ import annotations

import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from core.tiler import create_tiler
import core.tiler as tiler_module
from core.gpu_runtime import GpuResidentImage


class CpuParallelCropTests(unittest.TestCase):
    def setUp(self):
        self.image = np.arange(120 * 160 * 3, dtype=np.uint8).reshape(120, 160, 3)

    def assert_same_tiles(self, serial, parallel):
        self.assertEqual(len(serial), len(parallel))
        self.assertGreater(len(serial), 1)
        for left, right in zip(serial, parallel):
            self.assertEqual(
                (left.tile_id, left.x, left.y, left.width, left.height,
                 left.row, left.col, left.metadata),
                (right.tile_id, right.x, right.y, right.width, right.height,
                 right.row, right.col, right.metadata),
            )
            np.testing.assert_array_equal(left.image, right.image)
            self.assertFalse(np.shares_memory(self.image, right.image))

    def test_grid_and_edge_tiles(self):
        config = {"mode": "grid", "width": 43, "height": 38, "overlap_x": 5, "overlap_y": 4}
        serial = list(create_tiler(config, crop_workers=1).iter_tiles(self.image))
        parallel = list(create_tiler(config, crop_workers=4).iter_tiles(self.image))
        self.assert_same_tiles(serial, parallel)

    def test_explicit_worker_setting_is_bounded(self):
        config = {"mode": "grid", "width": 32, "height": 32, "crop_workers": 3}
        with mock.patch("core.tiler.os.cpu_count", return_value=4):
            self.assertEqual(create_tiler(config).crop_workers, 3)
            self.assertEqual(create_tiler(config, crop_workers=8).crop_workers, 4)
            self.assertEqual(create_tiler(config, crop_workers="invalid").crop_workers, 1)

    def test_auto_default_keeps_small_crops_serial(self):
        with mock.patch("core.tiler.ThreadPoolExecutor") as executor:
            tiles = list(create_tiler(
                {"mode": "grid", "width": 32, "height": 32},
            ).iter_tiles(self.image))
        self.assertEqual(create_tiler({"mode": "grid", "width": 32, "height": 32}).crop_workers, "auto")
        self.assertGreater(len(tiles), 1)
        executor.assert_not_called()

    def test_auto_default_uses_four_workers_for_large_crop_batch(self):
        image = np.zeros((2048, 2048, 3), np.uint8)
        with mock.patch("core.tiler.os.cpu_count", return_value=16), \
             mock.patch("core.tiler.ThreadPoolExecutor", wraps=ThreadPoolExecutor) as executor:
            tiles = list(create_tiler(
                {"mode": "grid", "width": 512, "height": 512},
            ).iter_tiles(image))
        self.assertEqual(len(tiles), 16)
        executor.assert_called_once_with(max_workers=4)

    def test_cpu_crops_run_on_multiple_threads(self):
        original_crop = tiler_module._crop_image
        barrier = threading.Barrier(2, timeout=5)
        worker_names = set()
        lock = threading.Lock()
        calls = 0

        def synchronized_crop(*args):
            nonlocal calls
            with lock:
                worker_names.add(threading.current_thread().name)
                calls += 1
                should_wait = calls <= 2
            if should_wait:
                barrier.wait()
            return original_crop(*args)

        with mock.patch("core.tiler.os.cpu_count", return_value=4), \
             mock.patch("core.tiler._crop_image", side_effect=synchronized_crop):
            tiles = list(create_tiler(
                {"mode": "grid", "width": 43, "height": 38}, crop_workers=4,
            ).iter_tiles(self.image))
        self.assertGreater(len(tiles), 2)
        self.assertGreaterEqual(len(worker_names), 2)

    def test_anchor_grid_matches_once_then_parallel_crops(self):
        rng = np.random.default_rng(7)
        template = rng.integers(0, 256, (12, 10, 3), dtype=np.uint8)
        image = np.zeros_like(self.image)
        image[20:32, 30:40] = template
        with tempfile.TemporaryDirectory() as temp_name:
            template_path = Path(temp_name) / "anchor.png"
            self.assertTrue(cv2.imwrite(str(template_path), template))
            config = {
                "mode": "grid", "width": 16, "height": 14,
                "overlap_x": 0, "overlap_y": 0,
                "template_path": str(template_path),
                "search_x": 0, "search_y": 0, "search_w": 80, "search_h": 70,
                "offset_x": 5, "offset_y": 6, "rows": 3, "cols": 4,
                "roi_w": 16, "roi_h": 14, "gap_x": 2, "gap_y": 3,
                "match_threshold": 0.9,
            }
            serial = list(create_tiler(config, crop_workers=1).iter_tiles(image))
            parallel = list(create_tiler(config, crop_workers=4).iter_tiles(image))
            resident = GpuResidentImage(object(), 1, image.shape[1], image.shape[0], 3)
            resident_tiles = list(create_tiler(config, resident_image=resident).iter_tiles(image))
        self.assertEqual((parallel[0].x, parallel[0].y), (35, 26))
        self.assertEqual((parallel[1].x, parallel[1].y), (53, 26))
        self.assert_same_tiles(serial, parallel)
        self.assertEqual(len(resident_tiles), len(serial))
        for expected, tile in zip(serial, resident_tiles):
            self.assertEqual((tile.x, tile.y, tile.width, tile.height),
                             (expected.x, expected.y, expected.width, expected.height))
            self.assertTrue(np.shares_memory(image, tile.image))
            np.testing.assert_array_equal(tile.image, expected.image)

    def test_contour_crops_preserve_accepted_order_and_metadata(self):
        config = {"mode": "contour", "shapes": {"crop_padding": 3}}
        contours = [np.array([[[0, 0]]], dtype=np.int32) for _ in range(4)]
        metadata = [
            {"bbox": [8 + i * 30, 12 + i * 20, 15, 12], "shape": "rectangle"}
            for i in range(4)
        ]

        def run(workers):
            tiler = create_tiler(config, crop_workers=workers)
            with mock.patch.object(tiler.segmenter, "make_mask", return_value=np.zeros(self.image.shape[:2], np.uint8)), \
                 mock.patch("core.tiler.cv2.findContours", return_value=(contours, None)), \
                 mock.patch.object(tiler.analyzer, "analyze", side_effect=metadata):
                return list(tiler.iter_tiles(self.image))

        self.assert_same_tiles(run(1), run(4))

    def test_pattern_match_crops_preserve_match_order(self):
        config = {"mode": "pattern_match", "pattern_match": {"crop_padding": 2}}
        matches = [
            {"x": 10 + i * 30, "y": 15 + i * 20, "width": 14, "height": 12, "score": 0.9}
            for i in range(4)
        ]

        def run(workers):
            tiler = create_tiler(config, crop_workers=workers)
            with mock.patch.object(tiler.matcher, "find_matches", return_value=matches):
                return list(tiler.iter_tiles(self.image))

        self.assert_same_tiles(run(1), run(4))

    def test_gpu_crop_remains_serial(self):
        runtime = mock.Mock(available=True, last_error=None)
        runtime.crop.side_effect = lambda image, x, y, w, h: image[y:y+h, x:x+w].copy()
        with mock.patch("core.tiler.ThreadPoolExecutor") as executor:
            tiles = list(create_tiler(
                {"mode": "grid", "width": 43, "height": 38},
                gpu_runtime=runtime, crop_workers=4,
            ).iter_tiles(self.image))
        self.assertGreater(len(tiles), 1)
        self.assertEqual(runtime.crop.call_count, len(tiles))
        executor.assert_not_called()

    def test_resident_grid_tiles_keep_cpu_views_for_fallback_and_reporting(self):
        resident = GpuResidentImage(object(), 1, self.image.shape[1], self.image.shape[0], 3)
        config = {"mode": "grid", "width": 43, "height": 38, "overlap_x": 5, "overlap_y": 4}
        reference = list(create_tiler(config, crop_workers=1).iter_tiles(self.image))
        with mock.patch("core.tiler._crop_image", side_effect=AssertionError("unexpected CPU copy")):
            tiles = list(create_tiler(config, resident_image=resident).iter_tiles(self.image))
        for expected, tile in zip(reference, tiles):
            self.assertEqual((tile.x, tile.y, tile.width, tile.height),
                             (expected.x, expected.y, expected.width, expected.height))
            self.assertTrue(np.shares_memory(self.image, tile.image))
            np.testing.assert_array_equal(tile.image, expected.image)
            self.assertEqual((tile.device_roi.x, tile.device_roi.y), (tile.x, tile.y))
        self.assertEqual(len(tiles), len(reference))


if __name__ == "__main__":
    unittest.main()
