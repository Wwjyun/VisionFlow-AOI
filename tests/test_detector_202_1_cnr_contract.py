"""Pin the two remaining 202-CS-SN-1 CNR properties that the GPU port must preserve.

1. The candidate order produced by ``_collect_candidates`` is "CNR descending".
   Python's ``list.sort`` is stable, so candidates with an exactly equal CNR keep
   the order in which the component labels were visited (ascending label =
   CLUSTER order of ``cv2.connectedComponentsWithStats``).  Nothing in the
   current suite pins that property, so a reimplementation could change the
   candidate order (and therefore the defect list order and the NG overlay draw
   order) while still passing an "equal set" comparison.

2. The reported per-candidate statistics are correct to float32 round-off.  The
   ring statistics are computed with ``np.mean``/``np.std`` over a
   boolean-indexed float32 array; the GPU port will use a different summation
   order, so the interesting question is how large the resulting CNR deviation
   is and whether it can move a PASS/NG decision.  The tests quantify it instead
   of assuming it is negligible.
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from detectors.detector_202_1 import Detector202_1


def _detector(**overrides) -> Detector202_1:
    params = {"center_mask_enabled": False, "edge_mask_enabled": False}
    params.update(overrides)
    return Detector202_1(params=params)


def _cnr_scene(seed: int, height: int = 320, width: int = 480) -> np.ndarray:
    """Deterministic scene with several defects of clearly different strength."""

    rng = np.random.default_rng(seed)
    base = np.full((height, width), 148, dtype=np.float64)
    # Low-frequency illumination drift so the Gaussian background is non-trivial.
    yy, xx = np.mgrid[0:height, 0:width]
    base += 9.0 * np.sin(xx / 97.0) + 6.0 * np.cos(yy / 61.0)
    base += rng.normal(0.0, 1.6, (height, width))
    gray = base
    for index, (y, x, size, delta) in enumerate(
        [
            (40, 60, 14, -70.0),
            (150, 120, 10, -34.0),
            (60, 300, 18, 55.0),
            (220, 360, 8, -95.0),
            (250, 90, 12, 26.0),
        ]
    ):
        gray[y : y + size, x : x + size] += delta + index
    return np.clip(gray, 0, 255).astype(np.uint8)


def _float64_ring_reference(
    image_float: np.ndarray,
    labels: np.ndarray,
    stats: np.ndarray,
    inclusion_mask: np.ndarray,
    label: int,
    height: int,
    width: int,
    params: dict,
) -> dict:
    """Recompute one candidate's ring statistics in float64.

    Same slicing/selection rules as ``Detector202_1._collect_candidates`` but with
    float64 accumulation, which is the reference the float32 result is compared
    against.
    """

    x, y, component_width, component_height, area = (
        int(value) for value in stats[label]
    )
    component_mask = (
        labels[y : y + component_height, x : x + component_width] == label
    )
    component_values = image_float[
        y : y + component_height, x : x + component_width
    ][component_mask]

    padding_min = int(params.get("background_padding_min_px", 8))
    padding_max = int(params.get("background_padding_max_px", 50))
    padding_scale = float(params.get("background_padding_scale", 1.5))
    pad = int(
        max(
            padding_min,
            min(
                padding_max,
                max(component_width, component_height) * padding_scale,
            ),
        )
    )
    x_start = max(0, x - pad)
    y_start = max(0, y - pad)
    x_stop = min(width, x + component_width + pad)
    y_stop = min(height, y + component_height + pad)

    local_image = image_float[y_start:y_stop, x_start:x_stop].astype(np.float64)
    local_labels = labels[y_start:y_stop, x_start:x_stop]
    local_inclusion = inclusion_mask[y_start:y_stop, x_start:x_stop]
    background_values = local_image[
        (local_labels != label) & local_inclusion
    ]
    min_background_pixels = int(params.get("min_background_pixels", 20))
    if background_values.size < min_background_pixels:
        background_values = image_float[inclusion_mask].astype(np.float64)

    defect_mean = float(np.mean(component_values.astype(np.float64)))
    background_mean = (
        float(np.mean(background_values)) if background_values.size else 0.0
    )
    background_std = (
        float(np.std(background_values)) if background_values.size else 0.0
    )
    contrast = abs(defect_mean - background_mean)
    cnr_noise_floor = float(params.get("cnr_noise_floor", 0.000001))
    return {
        "bbox": (x, y, component_width, component_height),
        "area": area,
        "defect_mean": defect_mean,
        "background_mean": background_mean,
        "background_std": background_std,
        "background_area": int(background_values.size),
        "contrast": contrast,
        "cnr": contrast / max(background_std, cnr_noise_floor),
    }


class Detector2021CnrOrderTests(unittest.TestCase):
    """The candidate/defect order contract."""

    def test_candidates_are_ordered_by_descending_cnr(self):
        gray = _cnr_scene(11)
        detector = _detector()
        analysis = detector._automatic_cnr_mask(gray)
        candidates = detector._collect_candidates(
            analysis["image_float"],
            analysis["candidate_mask"],
            analysis["inclusion_mask"],
        )
        self.assertGreater(len(candidates), 2)
        cnrs = [candidate.cnr for candidate in candidates]
        self.assertEqual(cnrs, sorted(cnrs, reverse=True))

    def test_cnr_ties_follow_ascending_component_label_order(self):
        """Exact CNR ties are broken by the component label visit order.

        The scene below contains three byte-identical defects placed in a
        constant background, so their ring statistics are identical and the
        stable ``list.sort`` must leave them in ascending label order, i.e. in
        increasing raster order of their bounding boxes.
        """

        height = width = 200
        gray = np.full((height, width), 150, dtype=np.uint8)
        for y, x in ((30, 30), (30, 110), (120, 70)):
            gray[y : y + 12, x : x + 12] = 40
        detector = _detector(background_kernel_size=31, min_background_pixels=20)
        analysis = detector._automatic_cnr_mask(gray)
        candidates = detector._collect_candidates(
            analysis["image_float"],
            analysis["candidate_mask"],
            analysis["inclusion_mask"],
        )
        self.assertEqual(len(candidates), 3)
        cnrs = {candidate.cnr for candidate in candidates}
        self.assertEqual(
            len(cnrs),
            1,
            "scene must produce an exact CNR tie, otherwise this contract is untested",
        )
        boxes = [candidate.bbox for candidate in candidates]
        self.assertEqual(
            boxes,
            sorted(boxes, key=lambda box: (box[1], box[0])),
            "equal-CNR candidates must stay in ascending component label order",
        )

    def test_defect_list_order_matches_candidate_order(self):
        gray = _cnr_scene(23)
        detector = _detector()
        analysis = detector._automatic_cnr_mask(gray)
        candidates = detector._collect_candidates(
            analysis["image_float"],
            analysis["candidate_mask"],
            analysis["inclusion_mask"],
        )
        defects = [
            detector._candidate_to_defect(candidate, analysis)
            for candidate in candidates
        ]
        self.assertEqual(
            [defect["bbox_local"] for defect in defects],
            [list(candidate.bbox) for candidate in candidates],
        )
        reported = [defect["metadata"]["cnr"] for defect in defects]
        self.assertEqual(reported, sorted(reported, reverse=True))


class Detector2021CnrPrecisionTests(unittest.TestCase):
    """Quantify the float32 ring-statistic deviation the GPU port must respect."""

    def test_ring_statistics_match_a_float64_recomputation(self):
        for seed, extra in (
            (101, {}),
            (102, {"min_background_pixels": 20}),
            (103, {"background_kernel_size": 21, "gaussian_sigma": 1.5}),
            (104, {"cnr_noise_floor": 0.5}),
            (105, {"morph_operation": "close", "morph_kernel": 5}),
            (106, {"connectivity": 4}),
        ):
            with self.subTest(seed=seed, extra=sorted(extra)):
                gray = _cnr_scene(seed)
                detector = _detector(**extra)
                analysis = detector._automatic_cnr_mask(gray)
                candidates = detector._collect_candidates(
                    analysis["image_float"],
                    analysis["candidate_mask"],
                    analysis["inclusion_mask"],
                )
                labels = None
                count, labels, stats, _ = cv2.connectedComponentsWithStats(
                    analysis["candidate_mask"],
                    connectivity=int(extra.get("connectivity", 8)),
                )
                self.assertGreater(count, 1)
                expected = {}
                for label in range(1, count):
                    x, y, component_width, component_height, area = (
                        int(value) for value in stats[label]
                    )
                    if area < analysis["min_area"]:
                        continue
                    expected[(x, y, component_width, component_height)] = (
                        _float64_ring_reference(
                            analysis["image_float"],
                            labels,
                            stats,
                            analysis["inclusion_mask"],
                            label,
                            gray.shape[0],
                            gray.shape[1],
                            detector.params,
                        )
                    )
                self.assertGreater(len(candidates), 0)
                for candidate in candidates:
                    reference = expected[candidate.bbox]
                    self.assertEqual(candidate.area, reference["area"])
                    self.assertEqual(
                        candidate.background_area, reference["background_area"]
                    )
                    for name in ("defect_mean", "background_mean", "background_std"):
                        actual = getattr(candidate, name)
                        target = reference[name]
                        error = abs(actual - target)
                        tolerance = 1e-5 * max(1.0, abs(target))
                        self.assertLessEqual(
                            error,
                            tolerance,
                            f"{name} deviates by {error} (actual={actual!r}, "
                            f"float64={target!r})",
                        )
                    contrast_error = abs(candidate.contrast - reference["contrast"])
                    self.assertLessEqual(
                        contrast_error, 1e-4 * max(1.0, reference["contrast"])
                    )
                    cnr_error = abs(candidate.cnr - reference["cnr"])
                    self.assertLessEqual(
                        cnr_error,
                        1e-4 * max(1.0, reference["cnr"]),
                        f"cnr deviates by {cnr_error} (actual={candidate.cnr!r}, "
                        f"float64={reference['cnr']!r})",
                    )

    def test_cnr_deviation_is_far_below_the_smallest_candidate_gap(self):
        """Measure whether a summation-order change could reorder candidates.

        ``Detector202_1.detect`` applies **no** CNR threshold: every candidate
        becomes a defect, so the CNR value never decides PASS/NG.  What it does
        decide is the candidate sort order, and that order is stable, so a
        candidate can only move if the float32/float64 deviation is at least as
        large as the gap to its neighbour.  This test measures both numbers and
        fails if the deviation ever approaches the gap.
        """

        worst_deviation = 0.0
        smallest_gap = float("inf")
        for seed in range(201, 209):
            gray = _cnr_scene(seed)
            detector = _detector()
            analysis = detector._automatic_cnr_mask(gray)
            candidates = detector._collect_candidates(
                analysis["image_float"],
                analysis["candidate_mask"],
                analysis["inclusion_mask"],
            )
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                analysis["candidate_mask"], connectivity=8
            )
            by_box = {}
            for label in range(1, count):
                x, y, component_width, component_height, area = (
                    int(value) for value in stats[label]
                )
                if area < analysis["min_area"]:
                    continue
                by_box[(x, y, component_width, component_height)] = (
                    _float64_ring_reference(
                        analysis["image_float"],
                        labels,
                        stats,
                        analysis["inclusion_mask"],
                        label,
                        gray.shape[0],
                        gray.shape[1],
                        detector.params,
                    )
                )
            for candidate in candidates:
                reference = by_box[candidate.bbox]
                worst_deviation = max(
                    worst_deviation, abs(candidate.cnr - reference["cnr"])
                )
            values = sorted(candidate.cnr for candidate in candidates)
            for first, second in zip(values, values[1:]):
                gap = second - first
                if gap > 0.0:
                    smallest_gap = min(smallest_gap, gap)

        self.assertGreater(worst_deviation, 0.0)
        # Record the measured bound explicitly.  A GPU port whose CNR deviation
        # exceeds 1e-3 would be within three orders of magnitude of the observed
        # gap and must be re-measured rather than assumed safe.
        self.assertLess(worst_deviation, 1e-3)
        if smallest_gap != float("inf"):
            self.assertLess(worst_deviation, smallest_gap)


if __name__ == "__main__":
    unittest.main()
