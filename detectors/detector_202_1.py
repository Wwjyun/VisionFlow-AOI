from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np

from core.parameter_schema import PARAMETER_GROUP_OUTER, specs_from_defaults
from core.preprocess_plan import Gray, PreprocessPlan
from detectors.detector_202 import Detector202


_MASK_PARAM_KEYS = (
    "center_mask_enabled",
    "center_mask_use_image_center",
    "center_mask_x",
    "center_mask_y",
    "center_mask_width",
    "center_mask_height",
    "edge_mask_enabled",
    "edge_inset_all",
    "edge_inset_left",
    "edge_inset_right",
    "edge_inset_top",
    "edge_inset_bottom",
)

_AUTO_CNR_DEFAULTS = {
    "background_kernel_size": 0,
    "background_kernel_divisor": 40,
    "background_kernel_min": 31,
    "background_kernel_max": 151,
    "gaussian_sigma": 0.0,
    "mad_scale": 1.4826,
    "noise_sigma_floor": 0.000001,
    "residual_threshold_floor": 8.0,
    "residual_sigma_multiplier": 3.0,
    "candidate_max_value": 255,
    "morph_operation": "open",
    "morph_kernel": 3,
    "morph_iterations": 1,
    "connectivity": 8,
    "min_component_area_px": 5,
    "min_component_area_ratio": 0.000001,
    "max_component_area_px": 0,
    "max_component_area_ratio": 0.05,
    "component_border_margin_px": 1,
    "background_padding_min_px": 8,
    "background_padding_max_px": 50,
    "background_padding_scale": 1.5,
    "min_background_pixels": 20,
    "cnr_noise_floor": 0.000001,
}


@dataclass(frozen=True, slots=True)
class _CnrCandidate:
    cnr: float
    contrast: float
    area: int
    bbox: tuple[int, int, int, int]
    defect_mean: float
    background_mean: float
    background_std: float
    background_area: int


class Detector202_1(Detector202):
    """Automatic CNR candidate detector based on AcceptanceChecker."""

    detector_id = "202-CS-SN-1"
    detector_name = "automatic_cnr_detector"
    display_name = "202-CS-SN-1 自動 CNR 偵測器"
    default_params = {
        **{key: Detector202.default_params[key] for key in _MASK_PARAM_KEYS},
        **_AUTO_CNR_DEFAULTS,
    }
    PARAM_SPEC = {
        **{key: Detector202.PARAM_SPEC[key] for key in _MASK_PARAM_KEYS},
        **specs_from_defaults(
            _AUTO_CNR_DEFAULTS,
            {
                "background_kernel_size": {
                    "minimum": 0,
                    "label": "Gaussian 背景核心",
                    "tooltip": "0 代表依影像尺寸自動計算；非 0 時必須為至少 3 的奇數。",
                },
                "background_kernel_divisor": {
                    "minimum": 1,
                    "label": "Gaussian 自動核心除數",
                },
                "background_kernel_min": {
                    "minimum": 3,
                    "odd": True,
                    "label": "Gaussian 自動核心下限",
                },
                "background_kernel_max": {
                    "minimum": 3,
                    "odd": True,
                    "label": "Gaussian 自動核心上限",
                },
                "gaussian_sigma": {"minimum": 0, "label": "Gaussian Sigma"},
                "mad_scale": {"minimum": 0, "label": "MAD 雜訊倍率"},
                "noise_sigma_floor": {
                    "minimum": 0.00000001,
                    "step": 0.000001,
                    "decimals": 8,
                    "label": "Robust Sigma 下限",
                },
                "residual_threshold_floor": {
                    "minimum": 0,
                    "label": "Residual 門檻下限",
                },
                "residual_sigma_multiplier": {
                    "minimum": 0,
                    "label": "Residual Sigma 倍率",
                },
                "candidate_max_value": {
                    "minimum": 1,
                    "maximum": 255,
                    "label": "候選遮罩最大值",
                },
                "morph_operation": {
                    "choices": ("none", "open", "close", "erode", "dilate"),
                    "label": "形態學操作",
                },
                "morph_kernel": {
                    "minimum": 1,
                    "odd": True,
                    "label": "形態學核心",
                },
                "morph_iterations": {"minimum": 0, "label": "形態學次數"},
                "connectivity": {
                    "choices": (4, 8),
                    "label": "Connected Components 連通性",
                },
                "min_component_area_px": {
                    "minimum": 0,
                    "parameter_group": PARAMETER_GROUP_OUTER,
                    "label": "候選最小面積 (px)",
                },
                "min_component_area_ratio": {
                    "minimum": 0,
                    "maximum": 1,
                    "step": 0.000001,
                    "decimals": 8,
                    "parameter_group": PARAMETER_GROUP_OUTER,
                    "label": "候選最小面積比例",
                },
                "max_component_area_px": {
                    "minimum": 0,
                    "parameter_group": PARAMETER_GROUP_OUTER,
                    "label": "候選最大面積 (px)",
                    "tooltip": "0 代表不套用固定像素上限。",
                },
                "max_component_area_ratio": {
                    "minimum": 0,
                    "maximum": 1,
                    "step": 0.001,
                    "decimals": 6,
                    "parameter_group": PARAMETER_GROUP_OUTER,
                    "label": "候選最大面積比例",
                    "tooltip": "0 代表不套用影像面積比例上限。",
                },
                "component_border_margin_px": {
                    "minimum": 0,
                    "parameter_group": PARAMETER_GROUP_OUTER,
                    "label": "候選邊界排除距離",
                },
                "background_padding_min_px": {
                    "minimum": 0,
                    "parameter_group": PARAMETER_GROUP_OUTER,
                    "label": "背景 Ring 最小外擴",
                },
                "background_padding_max_px": {
                    "minimum": 0,
                    "parameter_group": PARAMETER_GROUP_OUTER,
                    "label": "背景 Ring 最大外擴",
                },
                "background_padding_scale": {
                    "minimum": 0,
                    "parameter_group": PARAMETER_GROUP_OUTER,
                    "label": "背景 Ring 尺寸倍率",
                },
                "min_background_pixels": {
                    "minimum": 1,
                    "label": "局部背景最少像素數",
                },
                "cnr_noise_floor": {
                    "minimum": 0.00000001,
                    "step": 0.000001,
                    "decimals": 8,
                    "label": "CNR 雜訊分母下限",
                },
            },
        ),
    }

    _REFERENCE_REPOSITORY = "https://github.com/Wwjyun/AcceptanceChecker"
    _REFERENCE_COMMIT = "117fce477744188b97659a035b031fe3bf874260"
    _MORPH_OPERATIONS = {
        "open": cv2.MORPH_OPEN,
        "close": cv2.MORPH_CLOSE,
        "erode": cv2.MORPH_ERODE,
        "dilate": cv2.MORPH_DILATE,
    }

    def detect(self, image) -> list[dict]:
        height, width = image.shape[:2]
        device = None
        if getattr(self, "_active_device_roi", None) is not None:
            with self.measure_detection_stage("device_cnr_candidates"):
                device = self._device_candidates(height, width)

        if device is not None:
            candidates, analysis = device
        else:
            with self.measure_detection_stage("preprocess"):
                gray = self._make_gray(image)

            with self.measure_detection_stage("automatic_cnr_mask"):
                analysis = self._automatic_cnr_mask(gray)

            with self.measure_detection_stage("connected_components_and_cnr"):
                candidates = self._collect_candidates(
                    analysis["image_float"],
                    analysis["candidate_mask"],
                    analysis["inclusion_mask"],
                )

        geometry_started = time.perf_counter()
        defects = [
            self._candidate_to_defect(candidate, analysis) for candidate in candidates
        ]
        self._detection_stage_durations["result_assembly"] = (
            time.perf_counter() - geometry_started
        )
        return defects

    def _make_gray(self, image: np.ndarray) -> np.ndarray:
        plan = self.cached_preprocess_plan(
            image,
            ("202-1_auto_cnr_gray",),
            lambda: PreprocessPlan(
                name="202-1_auto_cnr_gray",
                operations=(Gray(),),
            ),
        )
        gray = self.execute_preprocess_plan(image, plan)
        self._record_debug_image("202-1_gray", gray)
        return gray

    def _exact_median(self, values: np.ndarray) -> float:
        """Median of a float32 array, matching `np.median` for float32 input.

        Uses the optional CUDA exact-median export when the runtime offers it. The device result is
        bit-exact against `np.median`, and anything else - a missing export, an older DLL, or a
        device error - falls back to `np.median` for the whole call, so a failed GPU step never
        produces a partially device-derived value.
        """
        runtime = getattr(self, "gpu_runtime", None)
        source = np.ascontiguousarray(values, dtype=np.float32)
        if (
            runtime is not None
            and getattr(runtime, "available", False)
            and getattr(runtime, "supports_exact_median", False)
            and self.use_gpu
        ):
            try:
                return float(runtime.median_f32(source))
            except Exception:
                pass
        if source is values and source.dtype == np.float32:
            return float(np.median(values))
        return float(np.median(source))

    def _background_blur(
        self,
        image_float: np.ndarray,
        kernel: int,
        sigma: float,
    ) -> tuple[np.ndarray, str]:
        """Gaussian background, on the device when the runtime can honour ``sigma``.

        Returns the background together with the backend that produced it, so the reports can say
        which one was used instead of leaving the reader to infer it.

        Uses the optional CUDA float32 Gaussian export.  The device filter is *not* bit-identical
        to ``cv2.GaussianBlur`` - the summation order differs - so the caller must treat the result
        as mathematically equivalent rather than byte-equal: the measured deviation is <= 4e-4 on a
        [0, 255] float32 operand and the candidate mask it produces is bit-identical across the
        whole 202 final-output matrix, while the four residual-derived diagnostics (``mad``,
        ``residual_median``, ``residual_threshold``, ``robust_noise_sigma``) drift in their last
        bits.

        A missing export, an older DLL that ignores ``sigma``, or any device error falls back to
        ``cv2.GaussianBlur`` for the whole call, so a failed GPU step never leaves the detector
        with a background from a different filter.
        """
        runtime = getattr(self, "gpu_runtime", None)
        if (
            runtime is not None
            and getattr(runtime, "available", False)
            and getattr(runtime, "supports_gaussian_blur_f32", False)
            and getattr(runtime, "supports_gaussian_f32_sigma", False)
            and self.use_gpu
        ):
            try:
                return runtime.gaussian_blur_f32(image_float, kernel, sigma), "cuda_f32"
            except Exception:
                pass
        return cv2.GaussianBlur(image_float, (kernel, kernel), sigma), "opencv_cpu"

    def _residual_statistics(
        self,
        image_float: np.ndarray,
        background: np.ndarray,
        residual: np.ndarray,
        candidate_max_value: int,
        mad_scale: float,
        noise_sigma_floor: float,
        residual_threshold_floor: float,
        residual_sigma_multiplier: float,
    ) -> tuple[np.ndarray, float, float, float, str]:
        """Residual central-moment threshold and candidate mask, on the device when possible.

        Returns ``(candidate_mask, residual_median, mad, residual_threshold, backend)``.

        The device path is one additive export, ``vf_cnr_mask_f32``: it builds the
        residual and its absolute deviation on the device from the image and background
        planes, computes both medians with the same bit-exact machinery as
        ``vf_median_f32``, evaluates the threshold in double exactly as the Python does,
        and compares in float32 (which is what NumPy does against a Python float).
        Measured agreement is exact on every compared field - ``residual_median`` and
        ``mad`` bit-exact, ``threshold`` double-exact and the mask byte-exact across the
        whole equivalence sweep - so unlike the Gaussian background this step adds no
        drift of its own.

        ``residual`` is passed in because the caller needs it for the debug overlay; the
        device path does not upload it, it only uses it on the CPU fallback.

        Any missing export or device error falls back to the NumPy reference for the
        whole step, so a failed GPU step never produces a partially device-derived mask.
        """

        runtime = getattr(self, "gpu_runtime", None)
        if (
            runtime is not None
            and getattr(runtime, "available", False)
            and getattr(runtime, "supports_cnr_mask_f32", False)
            and self.use_gpu
        ):
            try:
                device = runtime.cnr_mask_f32(
                    image_float,
                    background,
                    sigma_multiplier=residual_sigma_multiplier,
                    threshold_floor=residual_threshold_floor,
                    absolute_floor=noise_sigma_floor,
                    mad_scale=mad_scale,
                    candidate_value=candidate_max_value,
                )
                expected_shape = (image_float.shape[0], image_float.shape[1])
                if device["mask"].shape == expected_shape:
                    return (
                        device["mask"],
                        float(device["residual_median"]),
                        float(device["mad"]),
                        float(device["threshold"]),
                        "cuda_f32",
                    )
            except Exception:
                pass

        residual_median = self._exact_median(residual)
        mad = self._exact_median(np.abs(residual - residual_median))
        robust_noise_sigma = float(max(mad_scale * mad, noise_sigma_floor))
        residual_threshold = float(
            max(
                residual_threshold_floor,
                residual_sigma_multiplier * robust_noise_sigma,
            )
        )
        candidate_mask = (
            (np.abs(residual - residual_median) > residual_threshold).astype(np.uint8)
            * candidate_max_value
        )
        return candidate_mask, residual_median, mad, residual_threshold, "numpy_cpu"

    def _automatic_cnr_mask(self, gray: np.ndarray) -> dict:
        image_float = gray.astype(np.float32)
        height, width = gray.shape[:2]
        background_kernel = self._background_kernel(height, width)
        gaussian_sigma = float(self.params.get("gaussian_sigma", 0.0))
        mad_scale = float(self.params.get("mad_scale", 1.4826))
        noise_sigma_floor = float(self.params.get("noise_sigma_floor", 0.000001))
        residual_threshold_floor = float(
            self.params.get("residual_threshold_floor", 8.0)
        )
        residual_sigma_multiplier = float(
            self.params.get("residual_sigma_multiplier", 3.0)
        )
        candidate_max_value = int(self.params.get("candidate_max_value", 255))
        residual = None
        resident_result = None
        runtime = getattr(self, "gpu_runtime", None)
        device_roi = getattr(self, "_active_device_roi", None)
        if (
            runtime is not None
            and getattr(runtime, "available", False)
            and getattr(runtime, "supports_cnr_mask_u8_roi", False)
            and self.use_gpu
            and not self.export_debug_images
            and device_roi is not None
            and (int(device_roi.height), int(device_roi.width)) == (height, width)
        ):
            try:
                resident_result = runtime.cnr_mask_u8_roi(
                    device_roi,
                    kernel_size=background_kernel,
                    sigma=gaussian_sigma,
                    sigma_multiplier=residual_sigma_multiplier,
                    threshold_floor=residual_threshold_floor,
                    absolute_floor=noise_sigma_floor,
                    mad_scale=mad_scale,
                    candidate_value=candidate_max_value,
                )
            except Exception:
                resident_result = None

        if resident_result is not None and resident_result["mask"].shape == (height, width):
            candidate_mask = resident_result["mask"]
            residual_median = float(resident_result["residual_median"])
            mad = float(resident_result["mad"])
            residual_threshold = float(resident_result["threshold"])
            background_backend = "cuda_resident_fused"
            mask_backend = "cuda_resident_fused"
        else:
            background, background_backend = self._background_blur(
                image_float, background_kernel, gaussian_sigma
            )
            residual = image_float - background
            (
                candidate_mask,
                residual_median,
                mad,
                residual_threshold,
                mask_backend,
            ) = self._residual_statistics(
                image_float,
                background,
                residual,
                candidate_max_value,
                mad_scale,
                noise_sigma_floor,
                residual_threshold_floor,
                residual_sigma_multiplier,
            )
        robust_noise_sigma = float(max(mad_scale * mad, noise_sigma_floor))
        morph_operation = str(self.params.get("morph_operation", "open")).lower()
        morph_kernel = int(self.params.get("morph_kernel", 3))
        morph_iterations = int(self.params.get("morph_iterations", 1))
        cv_morphology = self._MORPH_OPERATIONS.get(morph_operation)
        if cv_morphology is not None and morph_iterations > 0 and morph_kernel > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (morph_kernel, morph_kernel)
            )
            if cv_morphology == cv2.MORPH_DILATE:
                candidate_mask = cv2.dilate(
                    candidate_mask, kernel, iterations=morph_iterations
                )
            elif cv_morphology == cv2.MORPH_ERODE:
                candidate_mask = cv2.erode(
                    candidate_mask, kernel, iterations=morph_iterations
                )
            else:
                candidate_mask = cv2.morphologyEx(
                    candidate_mask,
                    cv_morphology,
                    kernel,
                    iterations=morph_iterations,
                )

        inclusion_mask = self._apply_exclusion_masks(
            np.full(gray.shape, 255, dtype=np.uint8)
        )
        candidate_mask = cv2.bitwise_and(candidate_mask, inclusion_mask)

        if residual is not None:
            self._record_debug_image(
                "202-1_residual_abs",
                np.clip(np.abs(residual - residual_median), 0, 255).astype(np.uint8),
            )
        self._record_debug_image("202-1_candidate_mask", candidate_mask)
        minimum_area, maximum_area = self._effective_component_area_limits(
            height, width
        )
        return {
            "image_float": image_float,
            "candidate_mask": candidate_mask,
            "inclusion_mask": inclusion_mask.astype(bool),
            "background_kernel": background_kernel,
            "background_backend": background_backend,
            "residual_backend": mask_backend,
            "gaussian_sigma": gaussian_sigma,
            "residual_median": residual_median,
            "mad": mad,
            "robust_noise_sigma": robust_noise_sigma,
            "residual_threshold": residual_threshold,
            "min_area": minimum_area,
            "max_area": maximum_area,
            "mask_shape": (height, width),
            "component_backend": "opencv_cpu",
        }

    _DEVICE_MORPHOLOGY_CODES = {"open": 0, "close": 1, "dilate": 2, "erode": 3}
    # RTX 3090 per-ROI detector time, device candidates vs the resident-mask + OpenCV route, default
    # ring padding: 256x256 0.45x, 512x512 0.47x, 1024x1024 0.96x, 2000x2000 2.36x, 12000x2000 5.23x.
    # Each ring window is scanned sequentially inside one device thread, which small ROIs never
    # amortise, so smaller ROIs keep the host route (see gpu/README.md).
    DEVICE_CANDIDATES_MIN_PIXELS = 1024 * 1024

    def _device_candidate_parameters(self, height: int, width: int) -> tuple[list[int], list[float]]:
        """Pack the host candidate semantics in the ``vf_cnr_candidates_u8_roi`` parameter layout."""
        morph_operation = str(self.params.get("morph_operation", "open")).lower()
        morph_kernel = int(self.params.get("morph_kernel", 3))
        morph_iterations = int(self.params.get("morph_iterations", 1))
        morph_code = self._DEVICE_MORPHOLOGY_CODES.get(morph_operation)
        if morph_code is None or morph_iterations <= 0 or morph_kernel <= 1:
            morph_code = -1
        geometry = self._exclusion_geometry(width, height)
        center = geometry["center"]
        insets = geometry["insets"]
        minimum_area, maximum_area = self._effective_component_area_limits(height, width)
        maximum_area_enabled = (
            int(self.params.get("max_component_area_px", 0)) > 0
            or float(self.params.get("max_component_area_ratio", 0.05)) > 0
        )
        int_params = [
            self._background_kernel(height, width),
            int(self.params.get("candidate_max_value", 255)),
            morph_code,
            morph_kernel,
            morph_iterations,
            1 if center is not None else 0,
            *(center if center is not None else (0, 0, 0, 0)),
            insets["top"],
            insets["bottom"],
            insets["left"],
            insets["right"],
            int(self.params.get("connectivity", 8)),
            minimum_area,
            maximum_area,
            1 if maximum_area_enabled else 0,
            int(self.params.get("component_border_margin_px", 1)),
            int(self.params.get("background_padding_min_px", 8)),
            int(self.params.get("background_padding_max_px", 50)),
            int(self.params.get("min_background_pixels", 20)),
        ]
        real_params = [
            float(self.params.get("gaussian_sigma", 0.0)),
            float(self.params.get("residual_sigma_multiplier", 3.0)),
            float(self.params.get("residual_threshold_floor", 8.0)),
            float(self.params.get("noise_sigma_floor", 0.000001)),
            float(self.params.get("mad_scale", 1.4826)),
            float(self.params.get("background_padding_scale", 1.5)),
        ]
        return int_params, real_params

    def _device_candidates(self, height: int, width: int):
        """Candidate extraction kept entirely on the device, or ``None`` to use the host path.

        ``vf_cnr_candidates_u8_roi`` returns the component boxes and the float32 np.mean/np.std
        values bit-exactly, so contrast, CNR and the ordering below are the host expressions applied
        to identical inputs. Any missing export, unsupported parameter, ring that needs the whole
        included image, or device error returns ``None`` and the unchanged host path runs instead.
        """
        runtime = getattr(self, "gpu_runtime", None)
        device_roi = getattr(self, "_active_device_roi", None)
        if (
            runtime is None
            or not getattr(runtime, "available", False)
            or not getattr(runtime, "supports_cnr_candidates_u8_roi", False)
            or not self.use_gpu
            or self.export_debug_images
            or device_roi is None
            or (int(device_roi.height), int(device_roi.width)) != (height, width)
            or height * width < self.DEVICE_CANDIDATES_MIN_PIXELS
        ):
            return None
        int_params, real_params = self._device_candidate_parameters(height, width)
        try:
            result = runtime.cnr_candidates_u8_roi(device_roi, int_params, real_params)
        except Exception:
            return None
        records = np.asarray(result["records"], dtype=np.int32)
        stats = np.asarray(result["stats"], dtype=np.float32)
        if records.ndim != 2 or records.shape[1] != 7 or stats.shape != (records.shape[0], 3):
            return None
        cnr_noise_floor = float(self.params.get("cnr_noise_floor", 0.000001))
        candidates = []
        for record, values in zip(records, stats):
            x, y, component_width, component_height, area, background_area, _status = (
                int(value) for value in record
            )
            defect_mean = float(values[0])
            background_mean = float(values[1])
            background_std = float(values[2])
            contrast = abs(defect_mean - background_mean)
            cnr = contrast / max(background_std, cnr_noise_floor)
            candidates.append(
                _CnrCandidate(
                    cnr=float(cnr),
                    contrast=float(contrast),
                    area=area,
                    bbox=(x, y, component_width, component_height),
                    defect_mean=defect_mean,
                    background_mean=background_mean,
                    background_std=background_std,
                    background_area=background_area,
                )
            )
        # Same total order as the host path. Components arrive in first-raster-pixel order; OpenCV
        # numbers 8-connected components in its own scan order, which only matters for candidates
        # equal in CNR and in bbox top-left, which the host ordering cannot separate either.
        candidates.sort(
            key=lambda candidate: (-candidate.cnr, candidate.bbox[1], candidate.bbox[0])
        )
        mad_scale = float(self.params.get("mad_scale", 1.4826))
        noise_sigma_floor = float(self.params.get("noise_sigma_floor", 0.000001))
        mad = float(result["mad"])
        minimum_area, maximum_area = self._effective_component_area_limits(height, width)
        analysis = {
            "background_kernel": int_params[0],
            "background_backend": "cuda_resident_fused",
            "residual_backend": "cuda_resident_fused",
            "component_backend": "cuda_resident",
            "gaussian_sigma": float(self.params.get("gaussian_sigma", 0.0)),
            "residual_median": float(result["residual_median"]),
            "mad": mad,
            "robust_noise_sigma": float(max(mad_scale * mad, noise_sigma_floor)),
            "residual_threshold": float(result["threshold"]),
            "min_area": minimum_area,
            "max_area": maximum_area,
            "mask_shape": (height, width),
        }
        return candidates, analysis

    def _collect_candidates(
        self,
        image_float: np.ndarray,
        candidate_mask: np.ndarray,
        inclusion_mask: np.ndarray,
    ) -> list[_CnrCandidate]:
        label_count, labels_raw, stats_raw, _ = cv2.connectedComponentsWithStats(
            candidate_mask,
            connectivity=int(self.params.get("connectivity", 8)),
        )
        return self._collect_candidates_with_labels(
            image_float,
            candidate_mask,
            inclusion_mask,
            np.asarray(labels_raw),
            np.asarray(stats_raw),
            int(label_count),
        )

    def _collect_candidates_with_labels(
        self,
        image_float: np.ndarray,
        candidate_mask: np.ndarray,
        inclusion_mask: np.ndarray,
        labels: np.ndarray,
        stats: np.ndarray | None = None,
        label_count: int | None = None,
    ) -> list[_CnrCandidate]:
        """Collect ring-CNR candidates from a supplied component label map.

        ``_collect_candidates`` computes the labels with
        ``cv2.connectedComponentsWithStats`` and delegates here.  Accepting the label
        map lets a caller supply labels produced by a different implementation - which
        is how ``tools/cnr_label_order_impact.py`` measures whether OpenCV's label
        *numbering* is observable in the detector output.
        """

        height, width = candidate_mask.shape[:2]
        minimum_area, maximum_area = self._effective_component_area_limits(
            height, width
        )
        maximum_area_enabled = (
            int(self.params.get("max_component_area_px", 0)) > 0
            or float(self.params.get("max_component_area_ratio", 0.05)) > 0
        )
        connectivity = int(self.params.get("connectivity", 8))
        if label_count is None or stats is None:
            label_count, labels_raw, stats_raw, _ = cv2.connectedComponentsWithStats(
                candidate_mask,
                connectivity=connectivity,
            )
            labels = np.asarray(labels_raw)
            stats = np.asarray(stats_raw)
        else:
            labels = np.asarray(labels)
            stats = np.asarray(stats)
        candidates = []

        for label in range(1, label_count):
            x, y, component_width, component_height, area = (
                int(value) for value in stats[label]
            )
            if area < minimum_area or (
                maximum_area_enabled and area > maximum_area
            ):
                continue
            border_margin = int(self.params.get("component_border_margin_px", 1))
            if (
                x <= border_margin
                or y <= border_margin
                or x + component_width >= width - border_margin
                or y + component_height >= height - border_margin
            ):
                continue

            component_mask = (
                labels[y : y + component_height, x : x + component_width] == label
            )
            component_values = image_float[
                y : y + component_height, x : x + component_width
            ][component_mask]
            if component_values.size == 0:
                continue

            padding_min = int(self.params.get("background_padding_min_px", 8))
            padding_max = int(self.params.get("background_padding_max_px", 50))
            padding_scale = float(self.params.get("background_padding_scale", 1.5))
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

            local_image = image_float[y_start:y_stop, x_start:x_stop]
            local_labels = labels[y_start:y_stop, x_start:x_stop]
            local_inclusion = inclusion_mask[y_start:y_stop, x_start:x_stop]
            background_values = local_image[
                (local_labels != label) & local_inclusion
            ]
            min_background_pixels = int(
                self.params.get("min_background_pixels", 20)
            )
            if background_values.size < min_background_pixels:
                background_values = image_float[inclusion_mask]

            defect_mean = float(np.mean(component_values))
            background_mean = (
                float(np.mean(background_values)) if background_values.size else 0.0
            )
            background_std = (
                float(np.std(background_values)) if background_values.size else 0.0
            )
            contrast = abs(defect_mean - background_mean)
            cnr_noise_floor = float(self.params.get("cnr_noise_floor", 0.000001))
            cnr = contrast / max(background_std, cnr_noise_floor)
            candidates.append(
                _CnrCandidate(
                    cnr=float(cnr),
                    contrast=float(contrast),
                    area=area,
                    bbox=(x, y, component_width, component_height),
                    defect_mean=defect_mean,
                    background_mean=background_mean,
                    background_std=background_std,
                    background_area=int(background_values.size),
                )
            )

        # Order by CNR descending, then by the component's bounding box in raster order.
        #
        # The tie-break used to be implicit: this is a stable sort, so candidates with an
        # exactly equal CNR stayed in the order their component labels were visited, which
        # tied the defect list order to OpenCV's label *numbering*.  A replacement
        # connected-components implementation that produces the same components with a
        # different numbering would then reorder the output.  Exact ties are real - a
        # regular array of identical parts ties almost every candidate (measured 433/435)
        # - so the tie-break is now explicit and depends only on geometry.  Bounding boxes
        # are disjoint for connected components, so this is a total order and the sort is
        # deterministic regardless of how components are numbered.
        #
        # Measured effect on production-shaped noisy surfaces: 0 of 121 candidates sit in
        # a tie group, so this changes nothing there.
        candidates.sort(
            key=lambda candidate: (-candidate.cnr, candidate.bbox[1], candidate.bbox[0])
        )
        return candidates

    def _candidate_to_defect(self, candidate: _CnrCandidate, analysis: dict) -> dict:
        return {
            "type": "202-1_auto_cnr_ng",
            "bbox_local": list(candidate.bbox),
            "area": float(candidate.area),
            "confidence": 1.0,
            "metadata": {
                "method": "automatic_cnr",
                "cnr": float(candidate.cnr),
                "contrast": float(candidate.contrast),
                "defect_mean": float(candidate.defect_mean),
                "background_mean": float(candidate.background_mean),
                "background_std": float(candidate.background_std),
                "background_area_px": int(candidate.background_area),
                "robust_noise_sigma": float(analysis["robust_noise_sigma"]),
                "residual_median": float(analysis["residual_median"]),
                "mad": float(analysis["mad"]),
                "residual_threshold": float(analysis["residual_threshold"]),
                "background_kernel": int(analysis["background_kernel"]),
                "background_backend": str(analysis["background_backend"]),
                "residual_backend": str(analysis["residual_backend"]),
                "background_precision_note": (
                    "background_backend=opencv_cpu 時背景與 OpenCV 逐位相同；"
                    "background_backend=cuda_f32 時 device 的加法順序與 OpenCV 不同，"
                    "候選遮罩與 PASS/NG 判定已實測完全相同，但 mad／residual_median／"
                    "residual_threshold／robust_noise_sigma 這四個殘差衍生診斷值會有"
                    "尾位（約 1e-5）差異。residual_backend 則不引入任何額外差異："
                    "cuda_f32（vf_cnr_mask_f32）的 residual_median／mad 逐位元相同、"
                    "residual_threshold double 完全相同、候選遮罩逐位元組相同；"
                    "cuda_resident_fused（vf_cnr_mask_u8_roi）與這條既有 GPU chain"
                    "逐位元相同，且不再傳輸 gray/background operands。"
                ),
                "background_kernel_config": {
                    "configured_size": int(
                        self.params.get("background_kernel_size", 0)
                    ),
                    "auto_divisor": int(
                        self.params.get("background_kernel_divisor", 40)
                    ),
                    "auto_min": int(self.params.get("background_kernel_min", 31)),
                    "auto_max": int(
                        self.params.get("background_kernel_max", 151)
                    ),
                },
                "gaussian_sigma": float(analysis["gaussian_sigma"]),
                "mad_scale": float(self.params.get("mad_scale", 1.4826)),
                "noise_sigma_floor": float(
                    self.params.get("noise_sigma_floor", 0.000001)
                ),
                "residual_threshold_floor": float(
                    self.params.get("residual_threshold_floor", 8.0)
                ),
                "residual_sigma_multiplier": float(
                    self.params.get("residual_sigma_multiplier", 3.0)
                ),
                "candidate_max_value": int(
                    self.params.get("candidate_max_value", 255)
                ),
                "morphology": {
                    "operation": str(
                        self.params.get("morph_operation", "open")
                    ).lower(),
                    "kernel": int(self.params.get("morph_kernel", 3)),
                    "iterations": int(self.params.get("morph_iterations", 1)),
                },
                "connectivity": int(self.params.get("connectivity", 8)),
                "minimum_area_px": int(analysis["min_area"]),
                "maximum_area_px": int(analysis["max_area"]),
                "component_border_margin_px": int(
                    self.params.get("component_border_margin_px", 1)
                ),
                "background_padding": {
                    "minimum_px": int(
                        self.params.get("background_padding_min_px", 8)
                    ),
                    "maximum_px": int(
                        self.params.get("background_padding_max_px", 50)
                    ),
                    "scale": float(
                        self.params.get("background_padding_scale", 1.5)
                    ),
                },
                "min_background_pixels": int(
                    self.params.get("min_background_pixels", 20)
                ),
                "cnr_noise_floor": float(
                    self.params.get("cnr_noise_floor", 0.000001)
                ),
                "center_mask_enabled": bool(
                    self.params.get("center_mask_enabled", True)
                ),
                "center_mask_half_extents": [
                    int(self.params.get("center_mask_width", 100)),
                    int(self.params.get("center_mask_height", 630)),
                ],
                "edge_mask_enabled": bool(
                    self.params.get("edge_mask_enabled", True)
                ),
                "effective_edge_insets": self._effective_edge_insets(
                    analysis["mask_shape"][1],
                    analysis["mask_shape"][0],
                ),
                "component_backend": str(analysis["component_backend"]),
                "mask_order": "automatic_cnr_mask_exclusion_components",
                "reference_repository": self._REFERENCE_REPOSITORY,
                "reference_commit": self._REFERENCE_COMMIT,
            },
        }

    @classmethod
    def validate_parameters(cls, params: dict, _model_registry=None) -> None:
        values = {**cls.default_params, **params}
        kernel_size = int(values["background_kernel_size"])
        if kernel_size and (kernel_size < 3 or kernel_size % 2 == 0):
            raise ValueError(
                "background_kernel_size must be 0 or an odd integer at least 3"
            )
        if int(values["background_kernel_min"]) > int(
            values["background_kernel_max"]
        ):
            raise ValueError(
                "background_kernel_min must not exceed background_kernel_max"
            )
        if int(values["background_padding_min_px"]) > int(
            values["background_padding_max_px"]
        ):
            raise ValueError(
                "background_padding_min_px must not exceed background_padding_max_px"
            )
        maximum_area_px = int(values["max_component_area_px"])
        if maximum_area_px and int(values["min_component_area_px"]) > maximum_area_px:
            raise ValueError(
                "min_component_area_px must not exceed max_component_area_px"
            )
        maximum_area_ratio = float(values["max_component_area_ratio"])
        if (
            maximum_area_ratio
            and float(values["min_component_area_ratio"]) > maximum_area_ratio
        ):
            raise ValueError(
                "min_component_area_ratio must not exceed max_component_area_ratio"
            )

    def _background_kernel(self, height: int, width: int) -> int:
        configured = int(self.params.get("background_kernel_size", 0))
        if configured > 0:
            return configured
        divisor = max(1, int(self.params.get("background_kernel_divisor", 40)))
        return self._safe_odd_kernel(
            min(height, width) // divisor,
            min_kernel=int(self.params.get("background_kernel_min", 31)),
            max_kernel=int(self.params.get("background_kernel_max", 151)),
        )

    def _effective_component_area_limits(
        self, height: int, width: int
    ) -> tuple[int, int]:
        image_area = int(height) * int(width)
        minimum_area = max(
            int(self.params.get("min_component_area_px", 5)),
            int(
                float(self.params.get("min_component_area_ratio", 0.000001))
                * image_area
            ),
        )
        maximum_candidates = []
        maximum_area_px = int(self.params.get("max_component_area_px", 0))
        maximum_area_ratio = float(
            self.params.get("max_component_area_ratio", 0.05)
        )
        if maximum_area_px > 0:
            maximum_candidates.append(maximum_area_px)
        if maximum_area_ratio > 0:
            maximum_candidates.append(int(maximum_area_ratio * image_area))
        maximum_area = min(maximum_candidates) if maximum_candidates else 0
        return minimum_area, maximum_area

    @staticmethod
    def _safe_odd_kernel(
        base: int,
        min_kernel: int = 31,
        max_kernel: int = 201,
    ) -> int:
        kernel = max(min_kernel, min(max_kernel, int(base)))
        return kernel + 1 if kernel % 2 == 0 else kernel
