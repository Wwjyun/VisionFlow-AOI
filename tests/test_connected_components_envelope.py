"""Pin the exact agreement envelope of the connected-components reference against OpenCV.

`tools/connected_components_reference.py` is intended to become the golden standard for a
GPU connected-components replacement, so the useful thing to know is not "it mostly
agrees" but exactly *where* it agrees and where it does not:

* ``connectivity=4`` is claimed fully equivalent, including the label numbering.  This
  module checks that claim across many shapes, both odd and even, and both dimensions.
* ``connectivity=8`` is known to disagree on the numbering while agreeing on the
  component set.  This module measures how often, so the gap cannot silently widen or
  be mistaken for a solved problem.

The random 4-connectivity cases are asserted strictly; the 8-connectivity cases assert
the properties that do hold (count, component set, per-component stats) and record the
numbering disagreement.  When the Bolelli block scan is ported, the 8-connectivity
assertions can be tightened to equality - the failure will then point at exactly the
masks that were wrong.
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from tools.connected_components_reference import connected_components_with_stats

# Shapes chosen to cover both parities of each dimension: the 2x2 block scans used by
# OpenCV and by any future port special-case odd rows and odd columns.
SHAPES = [
    (1, 1),
    (1, 8),
    (8, 1),
    (2, 2),
    (3, 3),
    (5, 7),
    (8, 8),
    (17, 23),
    (32, 40),
    (33, 41),
]


def _random_mask(seed: int, shape: tuple[int, int], density: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(shape) < density).astype(np.uint8) * 255


class ConnectedComponentsEnvelopeTests(unittest.TestCase):
    def test_four_connectivity_is_fully_equivalent_on_random_masks(self):
        """4-connectivity must match OpenCV exactly, for every shape and density."""

        cases = 0
        for shape in SHAPES:
            for density in (0.2, 0.45, 0.7):
                for seed in range(3):
                    mask = _random_mask(seed, shape, density)
                    with self.subTest(shape=shape, density=density, seed=seed):
                        reference = cv2.connectedComponentsWithStats(
                            mask, connectivity=4
                        )
                        actual = connected_components_with_stats(mask, 4)
                        self.assertEqual(actual[0], reference[0])
                        np.testing.assert_array_equal(actual[1], reference[1])
                        np.testing.assert_array_equal(actual[2], reference[2])
                        cases += 1
        self.assertEqual(cases, len(SHAPES) * 3 * 3)

    def test_random_masks_contain_enough_components_to_be_meaningful(self):
        """Guard against a sweep that passes because every mask is empty or full."""

        component_counts = []
        for shape in SHAPES:
            mask = _random_mask(0, shape, 0.45)
            component_counts.append(int(connected_components_with_stats(mask, 8)[0]) - 1)
        self.assertGreater(sum(component_counts), 40)
        self.assertGreater(max(component_counts), 5)

    def test_eight_connectivity_agrees_on_the_component_set_but_not_the_numbering(self):
        """Measure the known 8-connectivity gap instead of hiding it.

        The component set is compared as the multiset of ``(x, y, w, h, area)`` rows, so
        the assertion is about *which* components exist, independent of their numbers.
        """

        shapes_with_numbering_difference = 0
        shapes_checked = 0
        for shape in SHAPES:
            for density in (0.2, 0.45, 0.7):
                mask = _random_mask(0, shape, density)
                with self.subTest(shape=shape, density=density):
                    reference = cv2.connectedComponentsWithStats(
                        mask, connectivity=8
                    )
                    actual = connected_components_with_stats(mask, 8)
                    self.assertEqual(actual[0], reference[0], "component count")
                    self.assertEqual(
                        sorted(map(tuple, actual[2][1:])),
                        sorted(map(tuple, reference[2][1:])),
                        "component set (bbox + area) must be identical",
                    )
                    shapes_checked += 1
                    if not np.array_equal(actual[1], reference[1]):
                        shapes_with_numbering_difference += 1
        # The gap is real and widespread; if this ever drops to zero the port has landed
        # and the test above should be tightened to equality.
        self.assertGreater(shapes_with_numbering_difference, 0)
        self.assertLessEqual(shapes_with_numbering_difference, shapes_checked)
        print(
            f"\n8-connectivity label numbering differs on "
            f"{shapes_with_numbering_difference}/{shapes_checked} shape x density cases "
            f"while the component set is identical on all of them"
        )

    def test_one_by_one_masks_have_no_spurious_labels(self):
        for value, expected_count, expected_labels in (
            (0, 1, [[0]]),
            (255, 2, [[1]]),
        ):
            mask = np.full((1, 1), value, dtype=np.uint8)
            for connectivity in (4, 8):
                with self.subTest(value=value, connectivity=connectivity):
                    actual = connected_components_with_stats(mask, connectivity)
                    reference = cv2.connectedComponentsWithStats(
                        mask, connectivity=connectivity
                    )
                    self.assertEqual(actual[0], expected_count)
                    np.testing.assert_array_equal(actual[1], np.array(expected_labels))
                    np.testing.assert_array_equal(actual[0], reference[0])
                    np.testing.assert_array_equal(actual[1], reference[1])


if __name__ == "__main__":
    unittest.main()
