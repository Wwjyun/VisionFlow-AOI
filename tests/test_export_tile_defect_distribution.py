from __future__ import annotations

import csv
import json
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
                    {"image_name": "a.png", "tile_id": "r0000_c0001", "detector_id": "401", "defect_type": "scratch", "area": "10", "area_unit": "px^2", "score": "0.8"},
                    {"image_name": "a.png", "tile_id": "r0000_c0001", "detector_id": "401", "defect_type": "dent", "area": "20", "area_unit": "px^2", "score": "0.7"},
                    {"image_name": "b.png", "tile_id": "r0000_c0001", "detector_id": "401", "defect_type": "scratch", "area": "30", "area_unit": "px^2", "score": "0.9"},
                    {"image_name": "b.png", "tile_id": "r0001_c0000", "detector_id": "202", "defect_type": "spot", "area": "40", "area_unit": "px^2", "score": "1"},
                    {"image_name": "b.png", "tile_id": "r0001_c0000", "detector_id": "401", "defect_type": "spot", "area": "50", "area_unit": "px^2", "score": "0.6"},
                ],
            )
            json_dir = Path(temporary) / "run" / "json"
            self._write_json_report(
                json_dir / "a.json",
                "a.png",
                [("r0000_c0001", "NG"), ("r0001_c0000", "PASS")],
            )
            self._write_json_report(
                json_dir / "b.json",
                "b.png",
                [("r0000_c0001", "PASS"), ("r0001_c0000", "NG")],
            )

            output_path, distribution = export_html_report(summary_path)

            self.assertEqual(output_path, Path(temporary) / "run" / DEFAULT_OUTPUT_NAME)
            self.assertEqual(distribution.total_defects, 5)
            self.assertEqual(distribution.tile_count, 2)
            self.assertEqual(distribution.image_count, 2)
            self.assertEqual(distribution.tiles["r0000_c0001"].defect_count, 3)
            self.assertEqual(distribution.tiles["r0000_c0001"].affected_image_count, 2)
            self.assertEqual(distribution.tiles["r0000_c0001"].defect_type_counts["scratch"], 2)
            self.assertEqual(distribution.rows_with_area, 5)
            self.assertEqual(distribution.rows_with_score, 5)
            self.assertEqual(len(distribution.inspection_records), 4)
            self.assertEqual(distribution.json_reports_scanned, 2)
            report = output_path.read_text(encoding="utf-8")
            self.assertIn("AOI Tile 缺陷互動分析報表", report)
            self.assertIn("r0000_c0001", report)
            self.assertIn("scratch × 2", report)
            self.assertIn("plotly.js v4.0.0", report)
            self.assertIn('id="tile-ng-heatmap"', report)
            self.assertIn('id="tile-ng-ranking"', report)
            self.assertIn('id="tile-type-heatmap"', report)
            self.assertIn('id="defect-treemap"', report)
            self.assertIn('id="top-image-rows"', report)
            self.assertIn("NG 率＝NG 次數 ÷ 檢測次數", report)

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

    def test_normalizes_optional_numbers_and_legacy_area_unit(self):
        with tempfile.TemporaryDirectory(prefix="visionflow_tile_distribution_") as temporary:
            summary_path = self._write_summary(
                Path(temporary) / "summary.csv",
                [
                    {"image_name": "a.png", "tile_id": "T1", "detector_id": "401", "defect_type": "scratch", "area": "12.5", "area_unit": "", "score": "0.75"},
                    {"image_name": "b.png", "tile_id": "T2", "detector_id": "401", "defect_type": "scratch", "area": "bad", "area_unit": "um^2", "score": "nan"},
                ],
            )

            distribution = load_summary_distribution(summary_path)

            self.assertEqual(distribution.records[0].area, 12.5)
            self.assertEqual(distribution.records[0].area_unit, "px^2")
            self.assertEqual(distribution.records[0].score, 0.75)
            self.assertIsNone(distribution.records[1].area)
            self.assertIsNone(distribution.records[1].score)
            self.assertEqual(distribution.invalid_area_rows, 1)
            self.assertEqual(distribution.invalid_score_rows, 1)

    def test_json_tile_denominator_can_derive_result_and_is_script_safe(self):
        with tempfile.TemporaryDirectory(prefix="visionflow_tile_distribution_") as temporary:
            root = Path(temporary) / "run"
            summary_path = self._write_summary(
                root / "csv" / "summary.csv",
                [{"image_name": "</script><b>", "tile_id": "r0000_c0000", "detector_id": "401", "defect_type": "scratch"}],
            )
            json_dir = root / "json"
            json_dir.mkdir(parents=True)
            (json_dir / "invalid.json").write_text("{", encoding="utf-8")
            payload = {
                "image_name": "</script><b>",
                "recipe_name": "R",
                "machine_id": "M",
                "product_id": "P",
                "final_result": "NG",
                "tiles": [
                    {"tile": {"tile_id": "r0000_c0000"}, "detectors": [{"pass": False}]},
                    {"tile": {"tile_id": "r0000_c0001"}, "detectors": [{"pass": True}]},
                ],
            }
            (json_dir / "valid.json").write_text(json.dumps(payload), encoding="utf-8")

            output_path, distribution = export_html_report(summary_path)

            self.assertEqual([record.tile_result for record in distribution.inspection_records], ["NG", "PASS"])
            self.assertEqual(distribution.json_parse_errors, 1)
            report = output_path.read_text(encoding="utf-8")
            self.assertNotIn("</script><b>", report)
            self.assertIn("\\u003c/script\\u003e\\u003cb\\u003e", report)

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
        fields = [
            "image_name",
            "recipe_name",
            "machine_id",
            "product_id",
            "final_result",
            "tile_id",
            "detector_id",
            "defect_type",
            "score",
            "area",
            "area_unit",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    @staticmethod
    def _write_json_report(
        path: Path,
        image_name: str,
        tile_results: list[tuple[str, str]],
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "image_name": image_name,
            "recipe_name": "RECIPE_A",
            "machine_id": "AOI_01",
            "product_id": "PRODUCT_A",
            "final_result": "NG" if any(result == "NG" for _, result in tile_results) else "PASS",
            "tiles": [
                {"tile": {"tile_id": tile_id}, "result": result, "detectors": []}
                for tile_id, result in tile_results
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
