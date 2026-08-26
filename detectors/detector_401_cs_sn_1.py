from __future__ import annotations

import time

import cv2
import numpy as np

from core.parameter_schema import (
    PARAMETER_GROUP_INNER,
    PARAMETER_GROUP_OUTER,
    specs_from_defaults,
)
from core.preprocess_plan import AdaptiveMean, Gray, PreprocessPlan
from detectors.base_detector import BaseDetector


class Detector401CsSn1(BaseDetector):
    """Adaptive-mean contour detector with configurable four-side masks."""

    detector_id = "401-CS-SN-1"
    detector_name = "adaptive_contour_detector"
    display_name = "401-CS-SN-1 adaptive contour detector"
    defect_type = "401_cs_sn_1_contour_ng"
    preprocess_plan_name = "401_cs_sn_1_preprocess"

    default_params = {
        "edge_mask_enabled": True,
        "edge_inset_all": 0,
        "edge_inset_left": 0,
        "edge_inset_right": 0,
        "edge_inset_top": 0,
        "edge_inset_bottom": 0,
        "adaptive_block_size": 156,
        "adaptive_c": -56.0,
        "max_value": 255,
        "binary_inv": False,
        "contour_mode": "list",
        "min_area": 0.0,
        "max_area": 0.0,
    }
    PARAM_SPEC = specs_from_defaults(
        default_params,
        {
            "edge_mask_enabled": {
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "啟用四邊屏蔽",
            },
            "edge_inset_all": {
                "minimum": 0,
                "parameter_group": PARAMETER_GROUP_OUTER,
                "label": "共同內縮",
            },
            "edge_inset_left": {
                "minimum": 0,
                "parameter_group": PARAMETER_GROUP_OUTER,
                "label": "左側內縮",
            },
            "edge_inset_right": {
                "minimum": 0,
                "parameter_group": PARAMETER_GROUP_OUTER,
                "label": "右側內縮",
            },
            "edge_inset_top": {
                "minimum": 0,
                "parameter_group": PARAMETER_GROUP_OUTER,
                "label": "上側內縮",
            },
            "edge_inset_bottom": {
                "minimum": 0,
                "parameter_group": PARAMETER_GROUP_OUTER,
                "label": "下側內縮",
            },
            "adaptive_block_size": {
                "minimum": 3,
                "maximum": 501,
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "自適應區塊（偶數自動加一）",
            },
            "adaptive_c": {
                "minimum": -255,
                "maximum": 255,
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "自適應二值化 C",
            },
            "max_value": {
                "minimum": 1,
                "maximum": 255,
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "二值化最大值",
            },
            "binary_inv": {
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "反相二值化",
            },
            "contour_mode": {
                "choices": ("external", "list", "tree", "ccomp"),
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "輪廓擷取模式",
            },
            "min_area": {
                "minimum": 0,
                "parameter_group": PARAMETER_GROUP_OUTER,
                "label": "最小面積",
            },
            "max_area": {
                "minimum": 0,
                "parameter_group": PARAMETER_GROUP_OUTER,
                "label": "最大面積",
            },
        },
    )

    _CONTOUR_MODES = {
        "external": cv2.RETR_EXTERNAL,
        "list": cv2.RETR_LIST,
        "tree": cv2.RETR_TREE,
        "ccomp": cv2.RETR_CCOMP,
    }

    def preprocess(self, image):
        return image

    def detect(self, image) -> list[dict]:
        with self.measure_detection_stage("preprocess"):
            binary = self._make_binary(image)
        with self.measure_detection_stage("find_contours"):
            contour_mode = str(self.params.get("contour_mode", "list")).lower()
            contours, _ = cv2.findContours(
                binary,
                self._CONTOUR_MODES.get(contour_mode, cv2.RETR_LIST),
                cv2.CHAIN_APPROX_SIMPLE,
            )

        geometry_started = time.perf_counter()
        defects = []
        height, width = image.shape[:2]
        configured_block = int(self.params.get("adaptive_block_size", 156))
        effective_block = self._effective_adaptive_block_size(configured_block)
        effective_insets = self._effective_edge_insets(width, height)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= 0.0 or not self._passes_area_filter(area):
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            defects.append(
                {
                    "type": self.defect_type,
                    "bbox_local": [int(x), int(y), int(box_width), int(box_height)],
                    "area": float(np.round(area, 3)),
                    "confidence": 1.0,
                    "metadata": {
                        "shape": "contour",
                        "threshold_method": (
                            "adaptive_mean_inv"
                            if bool(self.params.get("binary_inv", False))
                            else "adaptive_mean"
                        ),
                        "adaptive_block_size": configured_block,
                        "effective_adaptive_block_size": effective_block,
                        "adaptive_c": float(self.params.get("adaptive_c", -56.0)),
                        "max_value": int(self.params.get("max_value", 255)),
                        "contour_mode": contour_mode,
                        "min_area": float(self.params.get("min_area", 0.0)),
                        "max_area": float(self.params.get("max_area", 0.0)),
                        "edge_mask_enabled": bool(
                            self.params.get("edge_mask_enabled", True)
                        ),
                        "effective_edge_insets": effective_insets,
                        "mask_order": "gray_adaptive_mean_edge_mask_contours",
                    },
                }
            )

        self._detection_stage_durations["geometry_analysis"] = (
            time.perf_counter() - geometry_started
        )
        defects.sort(
            key=lambda item: (
                -item["area"],
                item["bbox_local"][1],
                item["bbox_local"][0],
            )
        )
        return defects

    def _make_binary(self, image: np.ndarray) -> np.ndarray:
        configured_block = int(self.params.get("adaptive_block_size", 156))
        effective_block = self._effective_adaptive_block_size(configured_block)
        adaptive_c = float(self.params.get("adaptive_c", -56.0))
        max_value = int(self.params.get("max_value", 255))
        binary_inv = bool(self.params.get("binary_inv", False))
        signature = (
            self.preprocess_plan_name,
            effective_block,
            adaptive_c,
            max_value,
            binary_inv,
        )
        plan = self.cached_preprocess_plan(
            image,
            signature,
            lambda: PreprocessPlan(
                name=self.preprocess_plan_name,
                operations=(
                    Gray(),
                    AdaptiveMean(
                        block_size=effective_block,
                        c=adaptive_c,
                        max_value=max_value,
                        invert=binary_inv,
                    ),
                ),
            ),
        )
        binary = self.execute_preprocess_plan(image, plan)
        self._record_debug_image("401-CS-SN-1_binary", binary)
        masked = self._apply_edge_mask(binary)
        self._record_debug_image("401-CS-SN-1_masked_binary", masked)
        return masked

    def _apply_edge_mask(self, binary: np.ndarray) -> np.ndarray:
        masked = binary.copy()
        if not bool(self.params.get("edge_mask_enabled", True)):
            return masked

        height, width = masked.shape[:2]
        insets = self._effective_edge_insets(width, height)
        if insets["top"] > 0:
            masked[: insets["top"], :] = 0
        if insets["bottom"] > 0:
            masked[height - insets["bottom"] :, :] = 0
        if insets["left"] > 0:
            masked[:, : insets["left"]] = 0
        if insets["right"] > 0:
            masked[:, width - insets["right"] :] = 0
        return masked

    def _effective_edge_insets(self, width: int, height: int) -> dict[str, int]:
        common = max(0, int(self.params.get("edge_inset_all", 0)))
        return {
            "left": min(
                max(common, max(0, int(self.params.get("edge_inset_left", 0)))),
                width,
            ),
            "right": min(
                max(common, max(0, int(self.params.get("edge_inset_right", 0)))),
                width,
            ),
            "top": min(
                max(common, max(0, int(self.params.get("edge_inset_top", 0)))),
                height,
            ),
            "bottom": min(
                max(common, max(0, int(self.params.get("edge_inset_bottom", 0)))),
                height,
            ),
        }

    def _passes_area_filter(self, area: float) -> bool:
        min_area = float(self.params.get("min_area", 0.0))
        max_area = float(self.params.get("max_area", 0.0))
        if min_area and area < min_area:
            return False
        if max_area and area > max_area:
            return False
        return True

    @staticmethod
    def _effective_adaptive_block_size(configured: int) -> int:
        block_size = max(3, int(configured))
        return block_size + 1 if block_size % 2 == 0 else block_size
