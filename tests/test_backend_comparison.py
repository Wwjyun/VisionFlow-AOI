from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import yaml
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from core.backend_comparison import BackendComparison, COMPARISON_OUTPUT_OVERRIDES, compare_decisions
from core.pipeline import AOIPipeline
from gui.backend_comparison_dialog import BackendComparisonDialog, comparison_headline
from gui.main_window import MainWindow

ROOT = Path(__file__).resolve().parents[1]


def _result(final="NG", area=40, confidence=0.9, active=True, end_to_end=0.2, detector_seconds=0.1):
    return {
        "recipe_name": "R",
        "final_result": final,
        "summary": {"defect_count": 1},
        "tiles": [{
            "tile": {"tile_id": "t0", "x": 0, "y": 0, "width": 10, "height": 10, "metadata": {}},
            "detectors": [{
                "detector_id": "202-CS-SN-1", "pass": False, "defect_count": 1,
                "defects": [{"type": "cnr", "bbox_local": [1, 2, 3, 4], "area": area, "confidence": confidence,
                             "metadata": {"cnr": 5.0, "mad": 1.0000001, "residual_backend": "cuda" if active else "cpu"}}],
            }],
        }],
        "execution": {
            "performance": {"end_to_end_sec": end_to_end, "stages_sec": {"image_load": 0.05, "detectors_total": detector_seconds},
                            "detectors_sec": {"202-CS-SN-1": detector_seconds}},
            "gpu": {"detectors": {"202-CS-SN-1": {"requested": True, "active": active,
                                                  "fallback_reason": "" if active else "CUDA DLL not found"}}},
        },
    }


class BackendComparisonCoreTests(unittest.TestCase):
    def test_runs_requested_then_forced_cpu_without_outputs_and_reports_stages(self):
        calls = []
        cpu = _result(active=False, end_to_end=5.0, detector_seconds=4.8)
        cpu["execution"]["gpu"] = {}
        gpu = _result(end_to_end=0.25, detector_seconds=0.08)
        gpu["tiles"][0]["detectors"][0]["defects"][0]["metadata"]["mad"] = 1.0000003

        def factory(recipe_path, output_dir, **kwargs):
            calls.append(kwargs)
            pipeline = Mock()
            pipeline.run.return_value = cpu if kwargs.get("gpu_mode_override") == "cpu" else gpu
            return pipeline

        session = object()
        progress = []
        summary = BackendComparison(factory).run(
            Path("r.yaml"), Path("i.bmp"), Path("out"), gpu_session=session,
            progress_callback=lambda percent, message: progress.append((percent, message)),
        )
        self.assertIs(calls[0]["gpu_session"], session)
        self.assertNotIn("gpu_mode_override", calls[0])
        self.assertEqual(calls[1]["gpu_mode_override"], "cpu")
        self.assertNotIn("gpu_session", calls[1], "the CPU reference never touches CUDA")
        for call in calls:
            self.assertEqual(call["output_overrides"], COMPARISON_OUTPUT_OVERRIDES)
        self.assertTrue(summary["gpu_active"])
        self.assertTrue(summary["decision_equal"], "only drifting diagnostics and backend provenance differ")
        self.assertEqual(summary["drift_counts"], {"mad": 1})
        total = summary["stages"][0]
        self.assertEqual((total["stage"], total["cpu_ms"], total["gpu_ms"], total["speedup"]), ("end_to_end", 5000.0, 250.0, 20.0))
        self.assertIn({"scope": "detector", "stage": "202-CS-SN-1", "cpu_ms": 4800.0, "gpu_ms": 80.0, "speedup": 60.0},
                      summary["stages"])
        self.assertEqual(progress[-1], (100, "CPU／GPU 對照完成"))

    def test_decision_mismatch_names_the_first_different_field(self):
        mismatch = compare_decisions(_result(area=40), _result(area=41))
        self.assertFalse(mismatch["decision_equal"])
        self.assertIn("40", mismatch["first_difference"]["cpu"])
        self.assertIn("41", mismatch["first_difference"]["gpu"])
        self.assertFalse(compare_decisions(_result(final="NG"), _result(final="PASS"))["decision_equal"])

    def test_headline_levels(self):
        base = {"stages": [{"stage": "end_to_end", "cpu_ms": 5000.0, "gpu_ms": 250.0, "speedup": 20.0}],
                "cpu_final": "NG", "gpu_final": "NG", "cpu_defects": 3, "gpu_defects": 3}
        text, level = comparison_headline({**base, "gpu_active": True, "decision_equal": True, "drift_counts": {}})
        self.assertEqual(level, "success")
        self.assertIn("判定欄位一致", text)
        self.assertIn("20.00×", text)
        text, level = comparison_headline({**base, "gpu_active": True, "decision_equal": False, "gpu_defects": 4})
        self.assertEqual(level, "warning")
        self.assertIn("不一致", text)
        text, level = comparison_headline({**base, "gpu_active": False, "gpu_fallback_reason": "CUDA DLL not found",
                                           "decision_equal": True})
        self.assertEqual(level, "warning")
        self.assertIn("未實際使用 CUDA（CUDA DLL not found）", text)


class GpuModeOverrideTests(unittest.TestCase):
    def test_cpu_override_runs_a_strict_cuda_recipe_on_cpu_without_touching_the_file(self):
        recipe = yaml.safe_load((ROOT / "recipes" / "PRODUCT_A_NEGATIVE_401_AOI_01.yaml").read_text(encoding="utf-8"))
        recipe["gpu"] = {"mode": "cuda", "dll_path": "missing/visionflow_cuda.dll", "fallback_to_cpu": False}
        recipe["detectors"]["401-AS-SN-1"]["use_gpu"] = True
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe_path = root / "strict.yaml"
            recipe_path.write_text(yaml.safe_dump(recipe, allow_unicode=True, sort_keys=False), encoding="utf-8")
            before = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
            image = np.full((600, 600, 3), 200, np.uint8)
            cv2.rectangle(image, (100, 100), (140, 130), (20, 20, 20), -1)
            image_path = root / "input.png"
            self.assertTrue(cv2.imwrite(str(image_path), image))

            with self.assertRaises(Exception):
                AOIPipeline(recipe_path, root / "strict", output_overrides=COMPARISON_OUTPUT_OVERRIDES).run(image_path)
            result = AOIPipeline(
                recipe_path, root / "cpu", output_overrides=COMPARISON_OUTPUT_OVERRIDES, gpu_mode_override="cpu"
            ).run(image_path)
            self.assertEqual(result["execution"]["gpu"]["mode"], "cpu")
            self.assertFalse(any(status.get("active") for status in result["execution"]["gpu"]["detectors"].values()))
            self.assertNotEqual(result["provenance"]["effective_recipe_sha256"], result["provenance"]["recipe_source_sha256"])
            self.assertEqual(hashlib.sha256(recipe_path.read_bytes()).hexdigest(), before)
        with self.assertRaises(ValueError):
            AOIPipeline(recipe_path, root, gpu_mode_override="cuda")


class BackendComparisonGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_engineer_comparison_uses_shared_cache_blocks_inspection_and_reports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            window = MainWindow(settings=QSettings(str(Path(temp_dir) / "compare.ini"), QSettings.Format.IniFormat))
            panel = window.run_screen.run_control_panel
            try:
                window._load_recipe(Path("recipes/PRODUCT_A_AOI_01.yaml"))
                window._start_preview_load = Mock()
                window.image_path = Path(temp_dir) / "input.bmp"
                window._update_run_ready()
                self.assertTrue(panel.compare_button.isEnabled())

                window.mode = "op"
                window.run_screen.set_mode("op")
                self.assertTrue(panel.isHidden(), "OP mode never sees the comparison")
                window._comparison_controller.start = Mock()
                window._run_backend_comparison()
                window._comparison_controller.start.assert_not_called()

                window.mode = "eng"
                window.run_screen.set_mode("eng")
                window._run_backend_comparison()
                worker = window._comparison_controller.start.call_args.args[0]
                self.assertIs(worker.gpu_session_cache, window._inspection_gpu_sessions)
                self.assertTrue(window.running)
                self.assertFalse(panel.start_button.isEnabled())
                self.assertEqual(panel.compare_button.text(), "CPU／GPU 對照中…")

                summary = {
                    "image_name": "input.bmp", "recipe_name": "R", "gpu_active": True, "decision_equal": True,
                    "drift_counts": {}, "cpu_final": "NG", "gpu_final": "NG", "cpu_defects": 2, "gpu_defects": 2,
                    "first_difference": None,
                    "stages": [{"scope": "total", "stage": "end_to_end", "cpu_ms": 5000.0, "gpu_ms": 250.0, "speedup": 20.0}],
                }
                window._on_backend_comparison_finished(summary)
                self.assertIn("判定欄位一致", window.notice_bar.label.text())
                dialog = window._comparison_dialog
                self.assertFalse(dialog.isModal())
                self.assertEqual(dialog.table.item(0, 1).text(), "端到端")
                self.assertEqual(dialog.table.item(0, 4).text(), "20.00×")
                dialog.close()

                window._on_backend_comparison_failed("strict CUDA failed")
                self.assertIn("CPU／GPU 對照失敗", window.notice_bar.label.text())
                window._on_backend_comparison_thread_finished()
                self.assertFalse(window.running)
                self.assertTrue(panel.compare_button.isEnabled())
                self.assertEqual(panel.compare_button.text(), "CPU／GPU 對照")
            finally:
                window._inspection_gpu_sessions.close()
                window.deleteLater()

    def test_dialog_lists_the_first_decision_difference(self):
        dialog = BackendComparisonDialog({
            "gpu_active": True, "decision_equal": False, "cpu_final": "NG", "gpu_final": "NG",
            "cpu_defects": 1, "gpu_defects": 1, "first_difference": {"index": 4, "cpu": "(40,)", "gpu": "(41,)"},
            "stages": [],
        })
        self.assertIn("第 5 項", dialog.difference_label.text())
        self.assertIn("不一致", dialog.headline_label.text())
        dialog.deleteLater()


if __name__ == "__main__":
    unittest.main()
