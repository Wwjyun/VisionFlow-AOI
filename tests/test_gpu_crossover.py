from __future__ import annotations

import unittest
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np

from core.gpu_crossover import PlanCrossoverPolicy
from core.gpu_runtime import GpuResidentImage, GpuRuntime
from core.pipeline_stages import InspectionResultAssembler
from core.preprocess_plan import Gray, PreprocessPlan
from detectors.detector_401 import Detector401
from tests.test_gpu_observability import _NativeDagPlanDll, _NativePlanDll

ROOT = Path(__file__).resolve().parents[1]


class _FixedCostPolicy(PlanCrossoverPolicy):
    """Replace wall-clock samples with deterministic per-backend costs."""

    def __init__(self, cuda_ms: float, cpu_ms: float, **kwargs):
        super().__init__(**kwargs)
        self.costs = {"cuda": cuda_ms / 1000.0, "cpu": cpu_ms / 1000.0}

    def record(self, key, backend, elapsed_sec):
        super().record(key, backend, self.costs[backend])


def _native_runtime(fallback_to_cpu: bool = True) -> tuple[GpuRuntime, _NativePlanDll]:
    runtime = GpuRuntime(enabled=False, fallback_to_cpu=fallback_to_cpu)
    dll = _NativePlanDll()
    runtime._dll = dll
    runtime.device_count = 1
    runtime._load_optional_context()
    return runtime, dll


def _detector_params() -> dict:
    return {
        "roi_inset_px": 0, "blur_size": 3, "morph_operation": "open", "morph_kernel": 3,
        "morph_iterations": 1, "adaptive_block_size": 3, "adaptive_c": 2.0, "min_area": 0, "max_area": 0,
    }


class PlanCrossoverPolicyTests(unittest.TestCase):
    def test_decision_waits_for_warmup_cuda_and_cpu_samples_then_freezes(self):
        policy = PlanCrossoverPolicy(cuda_samples=2, cpu_samples=1, margin=1.2)
        plan = PreprocessPlan((Gray(),), name="crossover_policy")
        key = policy.key(plan, np.zeros((8, 9, 3), dtype=np.uint8), device_roi=True)

        policy.record(key, "cuda", 5.0)  # warm-up includes native plan creation
        self.assertTrue(policy.wants_cpu_sample(key))
        policy.record(key, "cpu", 0.001)
        policy.record(key, "cuda", 0.002)
        self.assertEqual(policy.decision(key), "")
        policy.record(key, "cuda", 0.002)

        self.assertEqual(policy.decision(key), "cpu")
        self.assertFalse(policy.wants_cpu_sample(key))
        policy.record(key, "cuda", 0.0)
        self.assertEqual(policy.decision(key), "cpu")
        self.assertEqual(policy.report(key)["cuda_median_ms"], 2.0)

    def test_margin_keeps_cuda_when_cpu_is_only_slightly_faster(self):
        policy = PlanCrossoverPolicy(cuda_samples=1, cpu_samples=1, margin=1.15)
        key = ("plan", (4, 4), "|u1", False)
        for backend, seconds in (("cuda", 1.0), ("cuda", 0.0011), ("cpu", 0.0010)):
            policy.record(key, backend, seconds)

        self.assertEqual(policy.decision(key), "cuda")

    def test_entries_are_bounded_least_recently_used(self):
        policy = PlanCrossoverPolicy(max_entries=2)
        for index in range(3):
            policy.record(("plan", index), "cuda", 0.1)

        self.assertEqual(policy.report(("plan", 0)), {"decision": ""})
        self.assertFalse(policy.wants_cpu_sample(("plan", 0)))
        self.assertTrue(policy.wants_cpu_sample(("plan", 2)))


class ResidentUploadCrossoverTests(unittest.TestCase):
    def _decide(self, policy, plan, image, device_roi: bool) -> bool:
        return policy.prefer_cpu_key(policy.key(plan, image, device_roi))[0]

    def test_resident_cpu_decision_covers_host_input_and_upload_skip_is_bounded(self):
        policy = PlanCrossoverPolicy(cuda_samples=1, cpu_samples=1, max_entries=2)
        plan = PreprocessPlan((Gray(),), name="resident_skip")
        image = np.zeros((6, 7, 3), dtype=np.uint8)
        resident_key = policy.key(plan, image, device_roi=True)
        for backend, seconds in (("cuda", 1.0), ("cuda", 0.004), ("cpu", 0.001)):
            policy.record(resident_key, backend, seconds)

        self.assertTrue(self._decide(policy, plan, image, True))
        self.assertTrue(self._decide(policy, plan, image, False))
        self.assertEqual(policy.prefer_cpu_key(policy.key(plan, image, False))[1], resident_key)
        for index in range(3):
            policy.mark_resident_upload_unneeded(("recipe", index))
        self.assertFalse(policy.resident_upload_unneeded(("recipe", 0)))
        self.assertTrue(policy.resident_upload_unneeded(("recipe", 2)))

    def test_host_cpu_decision_does_not_imply_resident_cpu(self):
        policy = PlanCrossoverPolicy(cuda_samples=1, cpu_samples=1)
        plan = PreprocessPlan((Gray(),), name="host_only")
        image = np.zeros((6, 7, 3), dtype=np.uint8)
        host_key = policy.key(plan, image, device_roi=False)
        for backend, seconds in (("cuda", 1.0), ("cuda", 0.004), ("cpu", 0.001)):
            policy.record(host_key, backend, seconds)

        self.assertTrue(self._decide(policy, plan, image, False))
        self.assertFalse(self._decide(policy, plan, image, True))

    def test_pipeline_skips_resident_upload_after_all_gpu_plans_choose_cpu(self):
        import tempfile
        from copy import deepcopy
        from unittest.mock import Mock

        import cv2

        from core.gpu_session import GpuExecutionSession
        from core.pipeline import AOIPipeline
        from tests.test_gpu_session import _ResidentRuntime

        class _CrossoverResidentRuntime(_ResidentRuntime):
            def __init__(self):
                super().__init__()
                self.crossover_policy = PlanCrossoverPolicy()

        class _CpuCrossoverDetector:
            detector_id = "401-AS-SN-1"
            detector_name = "fake"
            display_name = "fake"
            use_gpu = True
            gpu_active = True
            gpu_fallback_reason = ""
            export_debug_images = False
            cpu_crossover_only = True
            preprocess_route_counts = {"cpu_crossover": 1}

            @staticmethod
            def cpu_crossover_covers(policy):
                return True

            def run(self, _image, device_roi=None, preprocess_cache=None):
                return {"detector_id": self.detector_id, "detector_name": "fake", "display_name": "fake",
                        "pass": True, "score": 0.0, "defects": [], "execution": {}}

        recipe_path = ROOT / "recipes" / "PRODUCT_A_NEGATIVE_401_AOI_01.yaml"
        recipe = deepcopy(AOIPipeline(recipe_path, ROOT / "outputs").recipe_manager.load(recipe_path))
        recipe["gpu"] = {"mode": "auto", "dll_path": "fake_resident.dll", "fallback_to_cpu": True, "tiling": False}
        recipe["detectors"]["401-AS-SN-1"]["use_gpu"] = True
        runtime = _CrossoverResidentRuntime()
        session = GpuExecutionSession(runtime, requested=True, config=recipe["gpu"])
        overrides = {key: False for key in ("save_overlay", "save_ng_tiles", "save_csv", "save_matrix_csv", "save_json")}
        reports = []
        with tempfile.TemporaryDirectory(prefix="visionflow_resident_skip_") as temporary:
            image_path = Path(temporary) / "input.png"
            self.assertTrue(cv2.imwrite(str(image_path), np.zeros((600, 600, 3), dtype=np.uint8)))
            for _ in range(3):
                pipeline = AOIPipeline(recipe_path, Path(temporary), output_overrides=overrides, gpu_session=session)
                pipeline.recipe_manager.load = Mock(return_value=recipe)
                pipeline.detector_manager.create_enabled = Mock(return_value=[_CpuCrossoverDetector()])
                reports.append(pipeline.run(image_path)["execution"]["gpu"]["resident_image"])

        self.assertEqual(runtime.upload_calls, 1)
        self.assertEqual([report["active"] for report in reports], [True, False, False])
        self.assertEqual([report["skipped_by_crossover"] for report in reports], [False, True, True])


class RealDetectorResidentUploadCrossoverTests(unittest.TestCase):
    """End-to-end: a real Detector's measured CPU route must also drop the whole-image H2D."""

    def _resident_runtime(self, cuda_ms: float, cpu_ms: float):
        runtime = GpuRuntime(enabled=False, fallback_to_cpu=True)
        dll = _NativeDagPlanDll()
        runtime._dll = dll
        runtime.device_count = 1
        runtime._load_optional_context()
        runtime.crossover_policy = _FixedCostPolicy(cuda_ms=cuda_ms, cpu_ms=cpu_ms, cuda_samples=1, cpu_samples=1)
        uploads = []
        real_upload = runtime.upload_image

        def counted_upload(image):
            uploads.append(int(image.nbytes))
            return real_upload(image)

        runtime.upload_image = counted_upload
        runtime.upload_bytes = uploads
        return runtime, dll

    def _run_pipeline(self, temporary: str, runtime, shapes=((512, 512),) * 3) -> tuple[list[dict], list]:
        """Run the real Detector401 through the pipeline once per requested image shape."""
        from unittest.mock import Mock

        import cv2

        from core.gpu_session import GpuExecutionSession
        from core.pipeline import AOIPipeline

        recipe_path = ROOT / "recipes" / "PRODUCT_A_NEGATIVE_401_AOI_01.yaml"
        recipe = deepcopy(AOIPipeline(recipe_path, ROOT / "outputs").recipe_manager.load(recipe_path))
        recipe["gpu"] = {"mode": "auto", "dll_path": "fake_resident.dll", "fallback_to_cpu": True, "tiling": False}
        recipe["detectors"]["401-AS-SN-1"]["use_gpu"] = True
        recipe["detectors"]["401-AS-SN-1"]["params"]["morph_iterations"] = 1
        recipe["detectors"]["401-AS-SN-1"]["params"]["max_area"] = 0
        recipe["detectors"]["401-AS-SN-1"]["params"]["min_area"] = 0
        session = GpuExecutionSession(runtime, requested=True, config=recipe["gpu"])
        overrides = {key: False for key in ("save_overlay", "save_ng_tiles", "save_csv", "save_matrix_csv", "save_json")}
        root = Path(temporary)
        image_path = root / "input.png"
        reports = []
        detectors = []
        for index, (height, width) in enumerate(shapes):
            image = np.zeros((height, width, 3), dtype=np.uint8)
            image[10:90, 20:120] = 200
            self.assertTrue(cv2.imwrite(str(image_path), image))
            detector = Detector401(params=_detector_params(), use_gpu=True, gpu_runtime=runtime)
            detectors.append(detector)
            pipeline = AOIPipeline(recipe_path, root, output_overrides=overrides, gpu_session=session)
            pipeline.recipe_manager.load = Mock(return_value=recipe)
            pipeline.detector_manager.create_enabled = Mock(return_value=[detector])
            reports.append(pipeline.run(image_path))
        return reports, detectors

    def test_measured_cpu_route_skips_resident_upload_from_the_next_image(self):
        import tempfile

        runtime, dll = self._resident_runtime(cuda_ms=0.30, cpu_ms=0.02)
        with tempfile.TemporaryDirectory(prefix="visionflow_real_skip_") as temporary:
            reports, detectors = self._run_pipeline(temporary, runtime, shapes=((512, 512),) * 4)

        # Image 1 runs the warm-up CUDA call and image 2 the measured CPU calibration sample; image 3
        # is the first run whose native GPU detector is wholly on the measured CPU route, which marks
        # this recipe/shape as upload-free. Image 4 therefore never uploads a pixel.
        self.assertEqual(len(runtime.upload_bytes), 3)
        self.assertGreaterEqual(dll.plan_roi_execute_calls, 2)
        self.assertEqual(dll.plan_execute_calls, 0)
        resident = [report["execution"]["gpu"]["resident_image"] for report in reports]
        self.assertEqual([item["active"] for item in resident], [True, True, True, False])
        self.assertEqual([item["skipped_by_crossover"] for item in resident], [False, False, False, True])
        self.assertTrue(runtime.crossover_policy.resident_upload_unneeded(
            (reports[0]["provenance"]["effective_recipe_sha256"], (512, 512, 3))
        ))
        self.assertEqual([detector.cpu_crossover_only for detector in detectors], [False, False, True, True])
        self.assertEqual(detectors[3].preprocess_route_counts.get("cuda"), None)
        # The last two images both run the measured CPU route and must stay result-identical.
        cpu_routed = reports[2:]
        self.assertEqual(cpu_routed[1]["final_result"], cpu_routed[0]["final_result"])
        self.assertEqual(cpu_routed[1]["summary"], cpu_routed[0]["summary"])
        self.assertEqual(
            [[tile["tile"]["tile_id"], tile["result"]] for tile in cpu_routed[1]["tiles"]],
            [[tile["tile"]["tile_id"], tile["result"]] for tile in cpu_routed[0]["tiles"]],
        )

    def test_measured_cuda_route_keeps_uploading_once_per_image(self):
        import tempfile

        runtime, dll = self._resident_runtime(cuda_ms=0.05, cpu_ms=2.0)
        with tempfile.TemporaryDirectory(prefix="visionflow_real_cuda_") as temporary:
            reports, detectors = self._run_pipeline(temporary, runtime)

        resident = [report["execution"]["gpu"]["resident_image"] for report in reports]
        self.assertEqual([item["active"] for item in resident], [True, True, True])
        self.assertEqual([item["skipped_by_crossover"] for item in resident], [False, False, False])
        self.assertEqual(len(runtime.upload_bytes), 3)
        self.assertGreaterEqual(dll.plan_roi_execute_calls, 3)
        self.assertEqual(dll.plan_execute_calls, 0)
        self.assertEqual(runtime.crossover_policy._resident_upload_unneeded, OrderedDict())
        self.assertTrue(all(not detector.cpu_crossover_only for detector in detectors))

    def test_unmeasured_image_shape_keeps_its_own_resident_upload(self):
        """A different image/tile shape has its own crossover key and must not inherit an earlier skip."""
        import tempfile

        runtime, dll = self._resident_runtime(cuda_ms=0.30, cpu_ms=0.02)
        with tempfile.TemporaryDirectory(prefix="visionflow_real_shape_") as temporary:
            reports, detectors = self._run_pipeline(
                temporary, runtime, shapes=((512, 512),) * 3 + ((512, 480),) + ((512, 512),)
            )

        resident = [report["execution"]["gpu"]["resident_image"] for report in reports]
        # Images 1-3 calibrate the 512x512 shape and mark its upload unnecessary, so image 5 reuses the
        # skip. The 512x480 image has its own unmeasured tile shape and must upload again.
        self.assertEqual([item["active"] for item in resident], [True, True, True, True, False])
        self.assertEqual(
            [item["skipped_by_crossover"] for item in resident], [False, False, False, False, True]
        )
        self.assertEqual(len(runtime.upload_bytes), 4)
        self.assertFalse(detectors[3].cpu_crossover_only)
        self.assertTrue(detectors[4].cpu_crossover_only)


class DetectorCrossoverRoutingTests(unittest.TestCase):
    def test_cheap_plan_moves_to_cpu_after_calibration_without_more_cuda_calls(self):
        runtime, dll = _native_runtime()
        runtime.crossover_policy = _FixedCostPolicy(cuda_ms=0.30, cpu_ms=0.05)
        image = np.random.default_rng(3).integers(0, 256, (32, 40, 3), dtype=np.uint8)
        reference = Detector401(params=_detector_params()).run(image)
        detector = Detector401(params=_detector_params(), use_gpu=True, gpu_runtime=runtime)

        calibration = [detector.run(image) for _ in range(4)]
        calls_after_calibration = dll.plan_execute_calls
        routed = [detector.run(image) for _ in range(3)]

        self.assertEqual(calls_after_calibration, 4)
        self.assertEqual(dll.plan_execute_calls, calls_after_calibration)
        self.assertTrue(all(result["execution"]["gpu_active"] for result in calibration))
        for result in routed:
            self.assertFalse(result["execution"]["gpu_active"])
            self.assertEqual(result["execution"]["backend"], "cpu")
            self.assertEqual(result["execution"]["fallback_reason"], "")
            self.assertEqual(result["execution"]["preprocess_capability"]["route"], "cpu_crossover")
            self.assertEqual(result["execution"]["preprocess_routes"], {"cpu_crossover": 1})
            self.assertEqual(result["defects"], reference["defects"])
        self.assertTrue(detector.gpu_active)
        self.assertFalse(detector.cpu_crossover_only)
        self.assertEqual(detector.preprocess_route_counts, {"cuda": 4, "cpu_crossover": 3})

    def test_expensive_plan_stays_on_cuda(self):
        runtime, dll = _native_runtime()
        runtime.crossover_policy = _FixedCostPolicy(cuda_ms=0.30, cpu_ms=2.0)
        detector = Detector401(params=_detector_params(), use_gpu=True, gpu_runtime=runtime)
        image = np.zeros((32, 40, 3), dtype=np.uint8)

        results = [detector.run(image) for _ in range(8)]

        self.assertEqual(dll.plan_execute_calls, 8)
        self.assertTrue(all(result["execution"]["preprocess_routes"] == {"cuda": 1} for result in results))

    def test_strict_cuda_runtime_has_no_crossover_policy(self):
        runtime, dll = _native_runtime(fallback_to_cpu=False)
        detector = Detector401(params=_detector_params(), use_gpu=True, gpu_runtime=runtime)

        for _ in range(8):
            detector.run(np.zeros((32, 40, 3), dtype=np.uint8))

        self.assertIsNone(runtime.crossover_policy)
        self.assertEqual(dll.plan_execute_calls, 8)

    def test_result_status_reports_crossover_only_detector_as_cpu_with_reason(self):
        runtime, _dll = _native_runtime()
        policy = _FixedCostPolicy(cuda_ms=0.30, cpu_ms=0.05)
        runtime.crossover_policy = policy
        image = np.zeros((32, 40, 3), dtype=np.uint8)
        warmup = Detector401(params=_detector_params(), use_gpu=True, gpu_runtime=runtime)
        for _ in range(4):
            warmup.run(image)
        detector = Detector401(params=_detector_params(), use_gpu=True, gpu_runtime=runtime)
        detector.run(image)

        status = InspectionResultAssembler._detector_gpu_status(detector, runtime)
        warm_status = InspectionResultAssembler._detector_gpu_status(warmup, runtime)

        self.assertTrue(detector.cpu_crossover_only)
        self.assertFalse(status["active"])
        self.assertEqual(status["backend"], "cpu")
        self.assertIn("CPU 較快", status["reason"])
        self.assertEqual(status["preprocess_routes"], {"cpu_crossover": 1})
        self.assertTrue(warm_status["active"])


if __name__ == "__main__":
    unittest.main()
