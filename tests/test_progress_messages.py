from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from core.batch_processor import BatchInspectionProcessor
from core.pipeline import AOIPipeline

ROOT = Path(__file__).resolve().parents[1]
NO_OUTPUTS = {key: False for key in ("save_overlay", "save_ng_tiles", "save_csv", "save_matrix_csv", "save_json")}
# Established industrial abbreviations and product terms that stay in English in operator text.
ALLOWED_WORDS = {"PASS", "NG", "CPU", "CUDA", "ROI", "DLL", "GPU", "Tile", "Detector", "Recipe", "overlay", "CSV", "JSON", "fallback", "worker"}
DETECTOR_ID = re.compile(r"\b\d{3}(?:-[A-Z]{2,4}){2}-\d\b")
CJK = re.compile(r"[一-鿿]")


def _english_words(message: str) -> set[str]:
    stripped = DETECTOR_ID.sub("", message)
    stripped = re.sub(r"\S+\.(?:png|bmp|jpe?g|tiff?)", "", stripped, flags=re.IGNORECASE)
    return set(re.findall(r"[A-Za-z]{2,}", stripped)) - ALLOWED_WORDS


class ProgressMessageLanguageTests(unittest.TestCase):
    def test_pipeline_and_batch_progress_messages_are_traditional_chinese(self):
        with tempfile.TemporaryDirectory(prefix="visionflow_progress_zh_") as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            image = np.full((700, 700, 3), 160, dtype=np.uint8)
            cv2.circle(image, (300, 300), 12, (20, 20, 20), -1)
            self.assertTrue(cv2.imwrite(str(images / "sample.png"), image))
            recipe = ROOT / "recipes" / "PRODUCT_A_AOI_01.yaml"

            messages: list[str] = []
            AOIPipeline(
                recipe, root / "out", output_overrides=NO_OUTPUTS,
                progress_callback=lambda _percent, message: messages.append(message),
            ).run(images / "sample.png")
            BatchInspectionProcessor(
                images, recipe, root / "batch", output_overrides=NO_OUTPUTS, max_workers=1,
                progress_callback=lambda _percent, message: messages.append(message),
            ).run()

        self.assertIn("開始檢測", messages)
        self.assertIn("檢測完成", messages)
        self.assertTrue(any(message.startswith("檢測 Tile ") for message in messages))
        self.assertTrue(any(message.startswith("批量 1/1：已完成 sample.png") for message in messages))
        for message in messages:
            with self.subTest(message=message):
                self.assertRegex(message, CJK)
                self.assertEqual(_english_words(message), set())


if __name__ == "__main__":
    unittest.main()
