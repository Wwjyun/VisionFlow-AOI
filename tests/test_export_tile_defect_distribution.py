from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from tools.export_tile_defect_distribution import (
    DEFAULT_OUTPUT_NAME,
    UNKNOWN_TILE_ID,
    default_output_path,
    export_html_report,
    load_summary_distribution,
    main,
)


class TileDefectDistributionTests(unittest.TestCase):
    def test_aggregates_summary_rows_by_tile_and_renders_heatmap(self):
        with tempfile.TemporaryDirectory(prefix="visionflow_tile_distribution_") as temporary:
            summary_path = self._write_summary(
                Path(temporary) / "run" / "csv" / "summary.csv",
                [
                    {"image_name": "a.png", "tile_id": "r0000_c0001", "detector_id": "401", "defect_type": "scratch"},
                    {"image_name": "a.png", "tile_id": "r0000_c0001", "detector_id": "401", "defect_type": "dent"},
                    {"image_name": "b.png", "tile_id": "r0000_c0001", "detector_id": "401", "defect_type": "scratch"},
                    {"image_name": "b.png", "tile_id": "r0001_c0000", "detector_id": "202", "defect_type": "spot"},
                    {"image_name": "b.png", "tile_id": "r0001_c0000", "detector_id": "401", "defect_type": "spot"},
                ],
            )

            output_path, distribution = export_html_report(summary_path)

            self.assertEqual(output_path, Path(temporary) / "run" / DEFAULT_OUTPUT_NAME)
            self.assertEqual(distribution.total_defects, 5)
            self.assertEqual(distribution.tile_count, 2)
            self.assertEqual(distribution.image_count, 2)
            self.assertEqual(distribution.tiles["r0000_c0001"].defect_count, 3)
            self.assertEqual(distribution.tiles["r0000_c0001"].affected_image_count, 2)
            self.assertEqual(distribution.tiles["r0000_c0001"].defect_type_counts["scratch"], 2)
            report = output_path.read_text(encoding="utf-8")
            self.assertIn("AOI Tile 缺陷分布報表", report)
            self.assertIn("r0000_c0001", report)
            self.assertIn("R0 C1", report)
            self.assertIn("scratch × 2", report)

    def test_handles_missing_tile_id_and_escapes_report_values(self):
        with tempfile.TemporaryDirectory(prefix="visionflow_tile_distribution_") as temporary:
            summary_path = self._write_summary(
                Path(temporary) / "summary.csv",
                [
                    {"image_name": "<image>", "tile_id": "", "detector_id": "<detector>", "defect_type": "<type>"},
                ],
            )

            output_path, distribution = export_html_report(summary_path)

            self.assertEqual(distribution.rows_without_tile_id, 1)
            self.assertIn(UNKNOWN_TILE_ID, distribution.tiles)
            report = output_path.read_text(encoding="utf-8")
            self.assertIn("&lt;detector&gt;", report)
            self.assertNotIn("<detector>", report)
            self.assertIn("未提供 tile_id", report)

    def test_empty_summary_still_creates_a_readable_html_report(self):
        with tempfile.TemporaryDirectory(prefix="visionflow_tile_distribution_") as temporary:
            summary_path = self._write_summary(Path(temporary) / "summary.csv", [])

            output_path, distribution = export_html_report(summary_path)

            self.assertEqual(distribution.total_defects, 0)
            self.assertTrue(output_path.is_file())
            self.assertIn("沒有缺陷資料列", output_path.read_text(encoding="utf-8"))

    def test_rejects_non_aoi_csv_and_can_run_in_cli_mode(self):
        with tempfile.TemporaryDirectory(prefix="visionflow_tile_distribution_") as temporary:
            root = Path(temporary)
            invalid = root / "invalid.csv"
            invalid.write_text("image_name,area\na.png,10\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tile_id"):
                load_summary_distribution(invalid)

            summary_path = self._write_summary(
                root / "summary.csv",
                [{"image_name": "a.png", "tile_id": "T1", "detector_id": "401", "defect_type": "scratch"}],
            )
            output_path = root / "custom-report.html"
            self.assertEqual(main(["--input", str(summary_path), "--output", str(output_path)]), 0)
            self.assertTrue(output_path.is_file())
            with self.assertRaisesRegex(ValueError, "不可覆寫"):
                export_html_report(summary_path, summary_path)

    @staticmethod
    def _write_summary(path: Path, rows: list[dict[str, str]]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["image_name", "tile_id", "detector_id", "defect_type"]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path


if __name__ == "__main__":
    unittest.main()
