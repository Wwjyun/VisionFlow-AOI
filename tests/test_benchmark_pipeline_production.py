import tempfile
import unittest
from pathlib import Path

from tools.benchmark_pipeline_production import (
    PRODUCTION_ANCHOR,
    PRODUCTION_BASE,
    PRODUCTION_CANVAS,
    PRODUCTION_GAP_X,
    PRODUCTION_ROI,
    PRODUCTION_ROI_ORIGINS,
    _recipe,
    _timing_summary,
)


class ProductionBenchmarkContractTests(unittest.TestCase):
    def test_user_confirmed_canvas_contains_six_tall_rois(self):
        self.assertEqual(PRODUCTION_CANVAS, (16384, 13000))
        self.assertEqual(PRODUCTION_ROI, (12000, 2000))
        self.assertEqual(len(PRODUCTION_ROI_ORIGINS), 6)
        for y, x in PRODUCTION_ROI_ORIGINS:
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + PRODUCTION_ROI[1], PRODUCTION_CANVAS[0])
            self.assertLessEqual(y + PRODUCTION_ROI[0], PRODUCTION_CANVAS[1])

    def test_production_recipe_uses_one_anchored_row_of_six_rois(self):
        with tempfile.TemporaryDirectory() as temporary:
            template = Path(temporary) / "anchor.png"
            recipe = _recipe(
                {"detectors": {}},
                "gpu/visionflow_cuda.dll",
                use_gpu=True,
                roi=PRODUCTION_ROI,
                template_path=template,
            )
        tile = recipe["tile"]
        self.assertEqual((tile["rows"], tile["cols"]), (1, 6))
        self.assertEqual((tile["roi_h"], tile["roi_w"]), PRODUCTION_ROI)
        self.assertEqual(tile["gap_x"], PRODUCTION_GAP_X)
        self.assertEqual(
            (tile["offset_x"], tile["offset_y"]),
            (
                PRODUCTION_BASE[0] - PRODUCTION_ANCHOR[0],
                PRODUCTION_BASE[1] - PRODUCTION_ANCHOR[1],
            ),
        )

    def test_timing_summary_uses_median_and_nearest_rank_p95(self):
        summary = _timing_summary([5.0, 1.0, 4.0, 2.0, 3.0])
        self.assertEqual(summary["median_ms"], 3.0)
        self.assertEqual(summary["p95_ms"], 5.0)
        self.assertEqual(summary["count"], 5)


if __name__ == "__main__":
    unittest.main()
