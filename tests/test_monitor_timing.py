from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from core.monitor_processor import FolderMonitorProcessor


class MonitorTimingTests(unittest.TestCase):
    def test_monitor_duration_includes_wait_pipeline_reports_and_move(self):
        pipeline = Mock()
        pipeline.run.return_value = {
            "final_result": "PASS",
            "summary": {"defect_count": 0, "ng_count": 0, "tile_count": 1},
            "duration_sec": 3.25,
            "outputs": {
                "overlay": "overlay/image.png",
                "csv": "csv/image.csv",
                "json": "json/image.json",
            },
            "execution": {"performance": {}},
            "tiles": [],
        }
        with tempfile.TemporaryDirectory(prefix="visionflow_monitor_timing_") as temporary:
            root = Path(temporary)
            source = root / "input" / "image.png"
            source.parent.mkdir()
            source.write_bytes(b"image")
            move_dir = root / "processed"
            processor = FolderMonitorProcessor(
                source.parent,
                root / "recipe.yaml",
                root / "output",
                processed_move_dir=move_dir,
            )

            with patch("core.monitor_processor.AOIPipeline", return_value=pipeline), patch(
                "core.monitor_processor.time.perf_counter",
                side_effect=[12.0, 16.0, 18.0, 18.5],
            ):
                result = processor._process_image(
                    source,
                    root / "output" / "monitor",
                    Mock(),
                    observed_at=10.0,
                    ready_at=11.0,
                )

        self.assertEqual(result.duration_sec, 8.5)
        self.assertEqual(
            result.timing,
            {
                "discovery_and_stability_wait_sec": 1.0,
                "queue_wait_sec": 1.0,
                "pipeline_and_reports_sec": 3.25,
                "processed_image_move_sec": 2.0,
                "end_to_end_sec": 8.5,
            },
        )
        self.assertEqual(result.moved_image_path, move_dir / "image.png")
        self.assertEqual(result.to_dict()["duration_sec"], 8.5)
        self.assertEqual(result.to_dict()["timing"], result.timing)

    def test_first_observation_survives_stability_checks_until_queueing(self):
        with tempfile.TemporaryDirectory(prefix="visionflow_monitor_stable_timing_") as temporary:
            root = Path(temporary)
            image_path = root / "image.png"
            image_path.write_bytes(b"image")
            processor = FolderMonitorProcessor(
                root,
                root / "recipe.yaml",
                root / "output",
                stable_checks=2,
            )
            with patch.object(processor, "_discover_images", return_value=[image_path]), patch.object(
                processor, "_estimate_arrival_monotonic", return_value=10.0
            ), patch("core.monitor_processor.time.time", return_value=100.0), patch(
                "core.monitor_processor.time.perf_counter", side_effect=[20.0, 21.0]
            ):
                processor._enqueue_new_stable_images()
                self.assertEqual(processor._pending, [])
                processor._enqueue_new_stable_images()

        self.assertEqual(processor._pending, [image_path])
        self.assertEqual(processor._observed_at[image_path], 10.0)
        self.assertEqual(processor._ready_at[image_path], 21.0)

    def test_arrival_timestamp_in_previous_poll_window_includes_discovery_wait(self):
        with tempfile.TemporaryDirectory(prefix="visionflow_monitor_arrival_") as temporary:
            root = Path(temporary)
            image_path = root / "image.png"
            image_path.write_bytes(b"image")
            birth_wall = float(
                getattr(image_path.stat(), "st_birthtime", image_path.stat().st_ctime)
            )
            processor = FolderMonitorProcessor(root, root / "recipe.yaml", root / "output")
            processor._last_scan_wall = birth_wall - 0.4
            processor._last_scan_monotonic = 50.0

            observed = processor._estimate_arrival_monotonic(
                image_path,
                scan_wall=birth_wall + 0.6,
                scan_monotonic=51.0,
            )

        self.assertAlmostEqual(observed, 50.4, places=3)


if __name__ == "__main__":
    unittest.main()
