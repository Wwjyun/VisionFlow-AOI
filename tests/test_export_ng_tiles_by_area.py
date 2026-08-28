from __future__ import annotations

import csv
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from tools.export_ng_tiles_by_area import (
    DEFAULT_RANGES,
    NgTileAreaClassifierWindow,
    TOOL_VERSION,
    UNMATCHED_FOLDER,
    classify_ng_tiles,
    parse_ranges,
    scan_ng_tiles,
)


class NgTileAreaClassifierTests(unittest.TestCase):
    def test_gui_constructs_with_default_ranges(self):
        app = QApplication.instance() or QApplication([])
        window = NgTileAreaClassifierWindow()
        self.assertEqual(window.windowTitle(), f"NG Tile 面積分類工具 v{TOOL_VERSION}")
        self.assertEqual(window.range_text.toPlainText(), DEFAULT_RANGES)
        self.assertTrue(window.include_unmatched.isChecked())
        window.close()
        app.processEvents()

    def test_parse_ranges_accepts_lines_and_rejects_overlap(self):
        ranges = parse_ranges("200-400\n401～500")
        self.assertEqual(
            [(item.lower, item.upper, item.folder_name) for item in ranges],
            [(200.0, 400.0, "200-400"), (401.0, 500.0, "401-500")],
        )
        with self.assertRaisesRegex(ValueError, "重疊"):
            parse_ranges("200-400,400-500")

    def test_scan_uses_selected_aggregation_and_legacy_px_unit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path, image_path = self._make_output(
                root,
                "run_a",
                rows=[
                    {"tile_id": "r0001_c0002", "area": "225"},
                    {"tile_id": "r0001_c0002", "area": "450"},
                ],
            )

            maximum = scan_ng_tiles(root, aggregation="max")
            total = scan_ng_tiles(root, aggregation="sum")

            self.assertEqual(maximum.csv_count, 1)
            self.assertEqual(maximum.records[0].area, 450.0)
            self.assertEqual(maximum.records[0].area_unit, "px^2")
            self.assertEqual(maximum.records[0].csv_path, csv_path)
            self.assertEqual(maximum.records[0].image_path, image_path)
            self.assertEqual(total.records[0].area, 675.0)

    def test_classifies_images_and_sidecars_without_moving_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, first = self._make_output(
                root,
                "run_a",
                rows=[{"tile_id": "T1", "area": "250", "area_unit": "px^2"}],
            )
            _, second = self._make_output(
                root,
                "run_b",
                rows=[{"tile_id": "T2", "area": "550", "area_unit": "px^2"}],
            )
            first.with_suffix(".json").write_text('{"review": "pending"}', encoding="utf-8")
            output = root / "classified"

            summary = classify_ng_tiles(
                root,
                output,
                parse_ranges("200-400\n401-500"),
            )

            self.assertEqual(summary.copied_count, 2)
            self.assertEqual(summary.unmatched_count, 1)
            self.assertTrue((output / "200-400" / first.name).is_file())
            self.assertTrue((output / "200-400" / first.with_suffix(".json").name).is_file())
            self.assertTrue((output / UNMATCHED_FOLDER / second.name).is_file())
            self.assertTrue(first.is_file())
            self.assertTrue(second.is_file())

    def test_mixed_units_are_separated_and_missing_images_are_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, px_image = self._make_output(
                root,
                "run_px",
                rows=[{"tile_id": "T1", "area": "300", "area_unit": "px^2"}],
            )
            _, um_image = self._make_output(
                root,
                "run_um",
                rows=[{"tile_id": "T2", "area": "300", "area_unit": "um^2"}],
            )
            csv_path, _ = self._make_output(
                root,
                "run_missing",
                rows=[{"tile_id": "T3", "area": "300", "area_unit": "px^2"}],
            )
            (csv_path.parent.parent / "ng_tiles" / f"{csv_path.stem}_T3.png").unlink()
            output = root / "classified"

            summary = classify_ng_tiles(root, output, parse_ranges("200-400"))

            self.assertEqual(summary.missing_image_count, 1)
            self.assertEqual(summary.units, ("px^2", "um^2"))
            self.assertTrue((output / "px2" / "200-400" / px_image.name).is_file())
            self.assertTrue((output / "um2" / "200-400" / um_image.name).is_file())

    @staticmethod
    def _make_output(
        root: Path,
        name: str,
        *,
        rows: list[dict[str, str]],
    ) -> tuple[Path, Path]:
        output = root / name
        csv_dir = output / "csv"
        ng_dir = output / "ng_tiles"
        csv_dir.mkdir(parents=True)
        ng_dir.mkdir(parents=True)
        csv_path = csv_dir / f"{name}.csv"
        fields = ["image_name", "tile_id", "area", "area_unit"]
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        tile_id = rows[0]["tile_id"]
        image_path = ng_dir / f"{name}_{tile_id}.png"
        image_path.write_bytes(f"image-{name}-{tile_id}".encode("utf-8"))
        return csv_path, image_path


if __name__ == "__main__":
    unittest.main()
