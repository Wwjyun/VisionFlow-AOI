from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QGraphicsView

from gui.image_viewer import ImageViewer
from gui.workers import ImagePreviewWorker


class GuiDisplayObservabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_preview_worker_reports_qimage_and_conversion_timings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "preview.png"
            encoded, buffer = cv2.imencode(".png", np.zeros((12, 16, 3), dtype=np.uint8))
            self.assertTrue(encoded)
            image_path.write_bytes(buffer.tobytes())
            loaded = []
            worker = ImagePreviewWorker(image_path, gpu_config={"display": False})
            worker.loaded.connect(lambda path, image, status: loaded.append((path, image, status)))

            worker.run()

        self.assertEqual(len(loaded), 1)
        path, image, status = loaded[0]
        self.assertEqual(path, image_path)
        self.assertFalse(image.isNull())
        snapshot = status["display_performance"]["worker"]
        self.assertIn("image_load", snapshot["stages_sec"])
        self.assertIn("color_conversion", snapshot["stages_sec"])
        self.assertIn("qimage_copy", snapshot["stages_sec"])
        self.assertGreaterEqual(snapshot["end_to_end_sec"], 0.0)

    def test_preview_color_conversion_never_loads_cuda_even_for_legacy_display_or_strict_recipes(self):
        from unittest.mock import patch

        from gui.workers import PREVIEW_DISPLAY_GPU_NOTE

        rng = np.random.default_rng(611)
        bgr = rng.integers(0, 256, size=(9, 13, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "preview.bmp"
            self.assertTrue(cv2.imwrite(str(image_path), bgr))
            configs = (
                {"display": False},
                {"mode": "auto", "display": True, "fallback_to_cpu": True},
                # Strict CUDA with a missing DLL used to fail the preview; display is CPU-only now.
                {"mode": "cuda", "display": True, "fallback_to_cpu": False, "dll_path": "missing/visionflow_cuda.dll"},
            )
            for config in configs:
                loaded, failed = [], []
                worker = ImagePreviewWorker(image_path, gpu_config=config)
                worker.loaded.connect(lambda path, image, status: loaded.append((image, status)))
                worker.failed.connect(lambda path, message: failed.append(message))
                with patch("core.gpu_runtime.GpuRuntime.__init__", side_effect=AssertionError("CUDA loaded")):
                    worker.run()
                self.assertEqual(failed, [], config)
                image, status = loaded[0]
                converted = image.convertToFormat(QImage.Format.Format_RGB888)
                pixels = np.frombuffer(converted.constBits(), dtype=np.uint8).reshape(9, converted.bytesPerLine())
                np.testing.assert_array_equal(pixels[:, : 13 * 3].reshape(9, 13, 3), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                self.assertEqual((status["requested"], status["active"], status["backend"]), (False, False, "cpu"))
                self.assertEqual(status.get("display_gpu_note", ""), PREVIEW_DISPLAY_GPU_NOTE if config["display"] else "")

        viewer = ImageViewer()
        viewer.set_backend_status({"requested": False, "active": False, "display_gpu_note": PREVIEW_DISPLAY_GPU_NOTE})
        self.assertEqual(viewer.backend_label.text(), "顯示: CPU")
        self.assertIn("不再生效", viewer.backend_label.toolTip())

    def test_viewer_reports_qpixmap_and_scene_timings_in_tooltip(self):
        image = QImage(16, 12, QImage.Format.Format_RGB888)
        image.fill(0)
        viewer = ImageViewer()

        snapshot = viewer.set_qimage(image, name="preview.png")
        status = {
            "requested": False,
            "active": False,
            "display_performance": {
                "worker": {"end_to_end_sec": 0.001},
                "viewer": snapshot,
                "user_wait_sec": 0.003,
            },
        }
        viewer.set_backend_status(status)

        self.assertIn("qpixmap_conversion", snapshot["stages_sec"])
        self.assertIn("scene_update", snapshot["stages_sec"])
        self.assertIn("fit_to_view", snapshot["stages_sec"])
        self.assertIn("QImage worker", viewer.backend_label.toolTip())
        self.assertIn("QPixmap/viewer", viewer.backend_label.toolTip())
        self.assertIn("User wait", viewer.backend_label.toolTip())

    def test_bulk_overlay_replacement_uses_reliable_bounded_repaint(self):
        image = QImage(640, 480, QImage.Format.Format_RGB888)
        image.fill(0)
        viewer = ImageViewer()
        viewer.resize(900, 640)
        viewer.show()
        viewer.set_qimage(image, name="overlay.png")
        overlays = [
            {
                "id": index,
                "tile_id": f"t{index}",
                "type": "tile_status",
                "bbox_global": [index % 600, index % 440, 20, 20],
                "score": 1.0,
                "status": "NG",
                "overlay_role": "tile_status",
            }
            for index in range(350)
        ]

        viewer.set_defects(overlays)
        self.app.processEvents()

        self.assertEqual(
            viewer.view.viewportUpdateMode(),
            QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate,
        )
        self.assertTrue(viewer.view.updatesEnabled())
        self.assertEqual(len(viewer._defect_items), 350)
        self.assertFalse(viewer.grab().isNull())

        viewer.set_defects(overlays[:3])
        self.app.processEvents()
        self.assertEqual(len(viewer._defect_items), 3)
        self.assertTrue(viewer.view.updatesEnabled())


if __name__ == "__main__":
    unittest.main()
