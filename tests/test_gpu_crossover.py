from __future__ import annotations

import unittest

import numpy as np

from core.gpu_crossover import PlanCrossoverPolicy
from core.gpu_runtime import GpuRuntime
from core.pipeline_stages import InspectionResultAssembler
from core.preprocess_plan import Gray, PreprocessPlan
from detectors.detector_401 import Detector401
from tests.test_gpu_observability import _NativePlanDll


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
