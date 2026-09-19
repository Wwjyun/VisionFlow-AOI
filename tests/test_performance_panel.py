from __future__ import annotations

import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QLabel

from gui.main_window import MainWindow, _backend_status_from_result
from gui.performance_summary import VRAM_LOW_NOTICE, performance_summary
from gui.screens.results_screen import ResultsScreen
from gui.widgets.topbar import TopBar


def _result(kind: str) -> dict:
    """Minimal inspection results shaped like real CUDA, CPU-only and fallback runs."""
    base = {
        "final_result": "NG",
        "duration_sec": 0.3,
        "summary": {"tile_count": 6, "ng_count": 6, "defect_count": 558},
        "tiles": [],
        "outputs": {},
        "execution": {
            "performance": {
                "stages_sec": {"image_load": 0.083, "initialization": 0.0588, "tiling": 0.0025,
                               "detectors_total": 0.0785, "reporting_total": 0.0013},
                "detector_stages_sec": {"202-CS-SN-1": {"device_cnr_candidates": 0.0517}},
                "end_to_end_sec": 0.2247,
            },
            "gpu": {"mode": "auto", "tiling": {"requested": False, "active": False}, "detectors": {}},
        },
    }
    gpu = base["execution"]["gpu"]
    if kind == "cuda":
        gpu["detectors"] = {"202-CS-SN-1": {"requested": True, "active": True, "device_name": "NVIDIA GeForce RTX 3090", "fallback_reason": ""}}
        gpu["device_host_split"] = {
            "image_decode": "cpu", "resident_upload": "device", "candidate_extraction": "device",
            "pass_ng_decision": "cpu", "hybrid_steps": {"candidate_extraction": ["vf_cnr_candidates_u8_roi"]},
            "note": "details",
        }
        gpu["metrics"] = {"call_count": 8, "host_to_device_bytes": 638980096, "device_to_host_bytes": 22340}
        gpu["resident_image"] = {
            "active": True, "skipped_by_crossover": False,
            "device_memory_before_upload": {"free_bytes": 22691184640, "total_bytes": 25769279488,
                                            "upload_bytes": 638976000, "dedicated_vram_low": False},
            "host_buffer": {"reused": True, "pinned": True, "decode_skipped": True, "nbytes": 638976000},
        }
    elif kind == "fallback":
        gpu["detectors"] = {"202-CS-SN-1": {"requested": True, "active": False, "fallback_reason": "CUDA DLL not found"}}
        gpu["device_host_split"] = {"image_decode": "cpu", "preprocessing": "cpu"}
        gpu["metrics"] = {"call_count": 0}
        gpu["resident_image"] = {"active": False, "device_memory_before_upload": {}}
    return base


class PerformanceSummaryTests(unittest.TestCase):
    def test_cuda_cpu_and_fallback_results_are_described_from_metadata(self):
        cuda = performance_summary(_result("cuda"))
        self.assertEqual(cuda["backend"], "CUDA · NVIDIA GeForce RTX 3090")
        self.assertEqual(cuda["total"], "225 ms")
        self.assertIn(("讀圖", "83.0 ms"), cuda["stages"])
        self.assertIn(("202-CS-SN-1 · device_cnr_candidates", "51.7 ms"), cuda["detector_stages"])
        self.assertIn(("候選抽取", "GPU（部分 CPU）"), cuda["split"])
        self.assertIn(("整圖上傳", "GPU"), cuda["split"])
        self.assertIn(("影像解碼", "CPU"), cuda["split"])
        transfer = dict(cuda["transfer"])
        self.assertEqual(transfer["整圖上傳"], "已上傳 609.4 MB")
        self.assertEqual(transfer["GPU→主機"], "21.8 KB")
        self.assertEqual(transfer["原生呼叫"], "8 次")
        self.assertEqual(transfer["主機影像緩衝"], "重用、pinned、略過讀檔")
        self.assertIn("可用 21.1 GB／總量 24.0 GB", transfer["顯示卡記憶體"])
        self.assertEqual((cuda["notices"], cuda["fallback_reasons"]), ([], []))

        cpu = performance_summary(_result("cpu"))
        self.assertEqual((cpu["backend"], cpu["backend_reason"], cpu["split"], cpu["transfer"]), ("CPU", "", [], []))

        fallback = performance_summary(_result("fallback"))
        self.assertEqual((fallback["backend"], fallback["backend_reason"]), ("CPU fallback", "CUDA DLL not found"))
        self.assertEqual(fallback["fallback_reasons"], ["202-CS-SN-1：CUDA DLL not found"])
        self.assertEqual(dict(fallback["transfer"])["整圖上傳"], "未上傳")

    def test_low_vram_and_crossover_skip_are_stated_in_text(self):
        result = _result("cuda")
        resident = result["execution"]["gpu"]["resident_image"]
        resident["device_memory_before_upload"].update(free_bytes=100, dedicated_vram_low=True)
        low = performance_summary(result)
        self.assertEqual(low["notices"], [VRAM_LOW_NOTICE])
        self.assertIn("（不足）", dict(low["transfer"])["顯示卡記憶體"])

        skipped = deepcopy(result)
        skipped["execution"]["gpu"]["resident_image"].update(active=False, skipped_by_crossover=True)
        self.assertIn("crossover 略過", dict(performance_summary(skipped)["transfer"])["整圖上傳"])


class PerformancePanelGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _panel_texts(screen: ResultsScreen) -> list[str]:
        return [label.text() for label in screen.performance_content.findChildren(QLabel)]

    def test_results_panel_populates_lazily_for_every_backend(self):
        screen = ResultsScreen()
        screen.set_result(_result("cuda"), None, "0.2 s")
        self.assertTrue(screen.performance_scroll.isHidden(), "collapsed by default so NG thumbnails keep their space")
        self.assertEqual(screen.performance_toggle.text(), "展開")
        screen.performance_toggle.click()
        self.assertFalse(screen.performance_scroll.isHidden())
        self.assertEqual(screen.performance_toggle.text(), "收合")
        for kind, expected in (("cuda", "GPU（部分 CPU）"), ("cpu", "CPU"), ("fallback", "CUDA DLL not found")):
            with self.subTest(kind=kind):
                screen.set_result(_result(kind), None, "0.2 s", defer_population=True)
                self.assertTrue(screen._content_dirty)
                screen.ensure_populated()
                texts = self._panel_texts(screen)
                self.assertIn("實際後端", texts)
                self.assertIn(expected, texts)
                self.assertEqual("傳輸與記憶體" in texts, kind != "cpu")
                self.assertFalse(screen.performance_scroll.isHidden(), "expanded state survives a new result")
        low = _result("cuda")
        low["execution"]["gpu"]["resident_image"]["device_memory_before_upload"]["dedicated_vram_low"] = True
        screen.set_result(low, None, "0.2 s")
        self.assertIn(f"注意：{VRAM_LOW_NOTICE}", self._panel_texts(screen))
        screen.deleteLater()

    def test_topbar_tooltip_and_window_notice_report_vram_state(self):
        topbar = TopBar()
        status = _backend_status_from_result(_result("cuda"))
        topbar.set_backend_status(status)
        self.assertIn("整圖上傳前可用顯示卡記憶體：21.1 GB／24.0 GB", topbar.backend_badge.toolTip())
        self.assertNotIn("注意", topbar.backend_badge.toolTip())

        low = _result("cuda")
        low["execution"]["gpu"]["resident_image"]["device_memory_before_upload"]["dedicated_vram_low"] = True
        topbar.set_backend_status(_backend_status_from_result(low))
        self.assertIn(VRAM_LOW_NOTICE, topbar.backend_badge.toolTip())
        topbar.set_backend_status(_backend_status_from_result(_result("cpu")))
        self.assertNotIn("顯示卡記憶體", topbar.backend_badge.toolTip())

        with tempfile.TemporaryDirectory() as temp_dir:
            window = MainWindow(settings=QSettings(str(Path(temp_dir) / "vram.ini"), QSettings.Format.IniFormat))
            try:
                window._on_inspection_finished(low)
                self.assertIn("效能提醒", window.notice_bar.label.text())
            finally:
                window._inspection_gpu_sessions.close()
                window.deleteLater()


if __name__ == "__main__":
    unittest.main()
