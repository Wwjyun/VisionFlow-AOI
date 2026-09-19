"""tools.contour_reference must reproduce cv2.findContours exactly.

Compared modes: cv2.RETR_LIST and cv2.RETR_EXTERNAL, both with cv2.CHAIN_APPROX_SIMPLE.
Every case checks contour count, per-contour shape, and np.array_equal of the point arrays
(so point order and coordinates are verified, not just the point set).
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from tools.contour_reference import find_contours

MODES = (("list", cv2.RETR_LIST), ("external", cv2.RETR_EXTERNAL))


def _points(contours):
    return [contour.reshape(-1, 2).tolist() for contour in contours]


def _mask_rect(height, width, y0, y1, x0, x1, value=255):
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y0:y1, x0:x1] = value
    return mask


class ContourReferenceTests(unittest.TestCase):
    """Reference tracer vs. OpenCV ground truth."""

    def assert_matches_opencv(self, mask, label):
        for mode_name, mode_code in MODES:
            reference = find_contours(mask, mode_name)
            expected, _ = cv2.findContours(mask, mode_code, cv2.CHAIN_APPROX_SIMPLE)
            context = f"{label} (mode={mode_name})"

            self.assertEqual(
                len(reference),
                len(expected),
                msg=(
                    f"{context}: contour count differs, reference={len(reference)} "
                    f"opencv={len(expected)}\n"
                    f"reference={_points(reference)}\nopencv={_points(expected)}"
                ),
            )

            for index, (actual, want) in enumerate(zip(reference, expected)):
                self.assertEqual(actual.dtype, np.int32, msg=f"{context}[{index}]: dtype")
                if actual.shape == want.shape and np.array_equal(actual, want):
                    continue
                actual_points = actual.reshape(-1, 2).tolist()
                want_points = want.reshape(-1, 2).tolist()
                first_diff = next(
                    (
                        position
                        for position in range(min(len(actual_points), len(want_points)))
                        if actual_points[position] != want_points[position]
                    ),
                    None,
                )
                self.fail(
                    f"{context}: contour {index} differs "
                    f"(reference shape={actual.shape}, opencv shape={want.shape}, "
                    f"first differing point index={first_diff})\n"
                    f"  reference={actual_points}\n"
                    f"  opencv   ={want_points}"
                )

    # ------------------------------------------------------------------ required masks

    def test_solid_rectangle(self):
        self.assert_matches_opencv(_mask_rect(12, 14, 3, 8, 4, 10), "solid rectangle")

    def test_two_separated_rectangles(self):
        mask = np.zeros((20, 24), dtype=np.uint8)
        mask[2:6, 3:8] = 255
        mask[12:18, 15:22] = 255
        self.assert_matches_opencv(mask, "two separated rectangles")

    def test_rectangle_with_hole(self):
        mask = _mask_rect(20, 20, 2, 17, 2, 17)
        mask[7:12, 7:12] = 0
        self.assert_matches_opencv(mask, "rectangle with a hole")

    def test_staircase_diagonal(self):
        mask = np.zeros((18, 18), dtype=np.uint8)
        for step in range(11):
            mask[2 + step, 2 + step] = 255
            mask[2 + step, 3 + step] = 255
            mask[3 + step, 2 + step] = 255
        self.assert_matches_opencv(mask, "staircase diagonal")

    def test_one_pixel_wide_line(self):
        horizontal = np.zeros((10, 12), dtype=np.uint8)
        horizontal[4, :] = 255
        self.assert_matches_opencv(horizontal, "1-pixel horizontal line")

        vertical = np.zeros((12, 10), dtype=np.uint8)
        vertical[:, 4] = 255
        self.assert_matches_opencv(vertical, "1-pixel vertical line")

        single = np.zeros((5, 5), dtype=np.uint8)
        single[2, 2] = 255
        self.assert_matches_opencv(single, "1-pixel dot")

    def test_shape_touching_the_image_border(self):
        corner = _mask_rect(12, 12, 0, 6, 0, 6)
        self.assert_matches_opencv(corner, "block touching the top-left corner")

        frame = np.zeros((14, 14), dtype=np.uint8)
        frame[0, :] = 255
        frame[-1, :] = 255
        frame[:, 0] = 255
        frame[:, -1] = 255
        self.assert_matches_opencv(frame, "image border frame")

        full_row = np.zeros((9, 9), dtype=np.uint8)
        full_row[0, :] = 255
        self.assert_matches_opencv(full_row, "top row only")

    def test_empty_image(self):
        empty = np.zeros((10, 12), dtype=np.uint8)
        self.assert_matches_opencv(empty, "empty image")
        self.assertEqual(find_contours(empty, "list"), [])
        self.assertEqual(find_contours(empty, "external"), [])

    def test_fully_filled_image(self):
        filled = np.full((10, 12), 255, dtype=np.uint8)
        self.assert_matches_opencv(filled, "fully filled image")

    def test_small_random_mask_after_morphological_opening(self):
        rng = np.random.default_rng(20240913)
        for trial in range(12):
            mask = (rng.random((14, 17)) < 0.45).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            self.assert_matches_opencv(mask, f"random mask #{trial} after MORPH_OPEN")

    # ------------------------------------------------------------------ extra coverage

    def test_seeded_random_battery(self):
        """Wider deterministic sweep so ring/termination edge cases stay covered."""
        rng = np.random.default_rng(7)
        for trial in range(120):
            height = int(rng.integers(3, 26))
            width = int(rng.integers(3, 26))
            density = float(rng.uniform(0.1, 0.9))
            mask = (rng.random((height, width)) < density).astype(np.uint8) * 255
            if trial % 3 == 0:
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            elif trial % 3 == 1:
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            self.assert_matches_opencv(mask, f"random battery #{trial}")

    def test_non_zero_pixels_of_any_value_are_foreground(self):
        mask = np.array(
            [
                [0, 1, 0, 200, 0],
                [7, 0, 255, 0, 3],
                [0, 0, 0, 0, 0],
                [9, 9, 0, 128, 0],
            ],
            dtype=np.uint8,
        )
        self.assert_matches_opencv(mask, "mixed non-zero label values")

    def test_nested_rings(self):
        mask = _mask_rect(20, 20, 2, 18, 2, 18)
        mask[5:15, 5:15] = 0
        mask[8:12, 8:12] = 255
        self.assert_matches_opencv(mask, "nested rings")

    def test_bool_and_non_contiguous_input(self):
        base = np.zeros((24, 24), dtype=np.uint8)
        base[5:19, 4:20] = 255
        base[9:14, 9:14] = 0
        self.assert_matches_opencv(base[3:22, 3:22], "non-contiguous view")

        boolean = base.astype(bool)
        reference = find_contours(boolean, "list")
        expected, _ = cv2.findContours(
            boolean.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        self.assertEqual(_points(reference), _points(expected))

    def test_output_shape_and_dtype(self):
        mask = _mask_rect(10, 10, 2, 8, 2, 8)
        for mode_name, _ in MODES:
            for contour in find_contours(mask, mode_name):
                self.assertEqual(contour.dtype, np.int32)
                self.assertEqual(contour.ndim, 3)
                self.assertEqual(contour.shape[1:], (1, 2))

    def test_reference_is_deterministic(self):
        rng = np.random.default_rng(4242)
        mask = (rng.random((22, 19)) < 0.5).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask[0:3, 0:3] = 255

        for mode_name, _ in MODES:
            first = find_contours(mask, mode_name)
            second = find_contours(mask, mode_name)
            self.assertEqual(len(first), len(second))
            for index, (left, right) in enumerate(zip(first, second)):
                self.assertTrue(
                    np.array_equal(left, right),
                    msg=f"mode={mode_name} contour {index} is not deterministic",
                )

        # The input mask must not be mutated by the tracer.
        before = mask.copy()
        for mode_name, _ in MODES:
            find_contours(mask, mode_name)
        self.assertTrue(np.array_equal(mask, before), "find_contours mutated its input")

    def test_unknown_mode_is_rejected(self):
        mask = _mask_rect(6, 6, 1, 4, 1, 4)
        with self.assertRaises(ValueError):
            find_contours(mask, "tree")


if __name__ == "__main__":
    unittest.main()
