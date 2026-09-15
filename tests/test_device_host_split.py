"""The result must report the actual device/host split of each pipeline step.

AGENT.md requires reporting what actually ran where instead of describing the flow as fully GPU,
so these tests pin the reported split for the CPU path, the resident-device path, and the case
where the crossover measured the CPU route.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from core.pipeline_stages import InspectionResultAssembler


class _Detector:
    def __init__(self, detector_id="401-AS-SN-1", use_gpu=True, gpu_active=True,
                 fallback="", routes=None, crossover=False):
        self.detector_id = detector_id
        self.use_gpu = use_gpu
        self.gpu_active = gpu_active
        self.gpu_fallback_reason = fallback
        self.preprocess_route_counts = dict(routes or {})
        self.cpu_crossover_only = crossover
        self.actual_backend = "cuda_dll" if gpu_active else "cpu"
        self.device_name = "fake"


class _Runtime:
    device_name = "fake"

    @staticmethod
    def status(requested=False):
        return {"requested": requested, "active": requested}


class _CountingRuntime(_Runtime):
    """A runtime that reports the optional device exports it actually executed."""

    def __init__(self, functions):
        self._functions = {
            name: {"calls": calls} for name, calls in dict(functions).items()
        }

    def performance_stats(self):
        return {
            "measurement_scope": "host_wrapper_and_optional_cuda_events",
            "functions": {
                name: {**entry, "host_to_device_bytes": 0, "device_to_host_bytes": 0}
                for name, entry in self._functions.items()
            },
        }


class DeviceHostSplitTests(unittest.TestCase):
    def _split(self, detectors, resident=True, tiling=False, runtime=None):
        return InspectionResultAssembler._device_host_split(
            gpu_runtime=runtime if runtime is not None else _Runtime(),
            detectors=detectors,
            resident_image=object() if resident else None,
            tiling_gpu_requested=tiling,
        )

    def test_cpu_only_run_reports_every_step_as_host(self):
        split = self._split([_Detector(use_gpu=False, gpu_active=False)], resident=False)
        for step in (
            "image_decode", "resident_upload", "anchor_localization", "tiling_roi",
            "preprocessing", "automatic_cnr_mask", "candidate_extraction", "geometry_and_statistics",
            "pass_ng_decision", "aggregation_and_reporting",
        ):
            self.assertEqual(split[step], "cpu", step)

    def test_resident_device_run_reports_device_steps_and_keeps_candidate_extraction_on_host(self):
        detector = _Detector(routes={"cuda": 6})
        split = self._split([detector], resident=True)
        self.assertEqual(split["image_decode"], "cpu")
        self.assertEqual(split["resident_upload"], "device")
        self.assertEqual(split["anchor_localization"], "device")
        self.assertEqual(split["preprocessing"], "device")
        # No device export for these steps ran in this run, so they must not be implied as device
        # work just because the recipe requested CUDA.
        self.assertEqual(split["candidate_extraction"], "cpu")
        self.assertEqual(split["automatic_cnr_mask"], "cpu")
        self.assertEqual(split["geometry_and_statistics"], "cpu")
        self.assertEqual(split["pass_ng_decision"], "cpu")
        self.assertEqual(split["hybrid_steps"], {})

    def test_executed_device_exports_mark_the_step_as_a_hybrid(self):
        """A step is ``device`` only when its device export really ran.

        ``Detector202_1`` computes its exact median/MAD on the device
        (``vf_median_f32``) while connected components still runs in OpenCV, so the
        candidate step is genuinely hybrid and must be reported as such, with the
        observed export named so the claim is checkable.
        """

        runtime = _CountingRuntime({"vf_median_f32": 12, "vf_context_upload_u8": 1})
        split = self._split([_Detector(routes={"cuda": 6})], resident=True, runtime=runtime)
        self.assertEqual(split["candidate_extraction"], "device")
        self.assertEqual(split["hybrid_steps"], {"candidate_extraction": ["vf_median_f32"]})
        self.assertIn("vf_median_f32", split["note"])
        # A step with no device export in this run stays host work.
        self.assertEqual(split["geometry_and_statistics"], "cpu")
        self.assertEqual(split["pass_ng_decision"], "cpu")

    def test_unknown_or_zero_counts_never_imply_device_work(self):
        for functions in ({}, {"vf_median_f32": 0}, {"vf_upload_unrelated_export": 5}):
            with self.subTest(functions=functions):
                runtime = _CountingRuntime(functions)
                split = self._split([_Detector()], resident=True, runtime=runtime)
                self.assertEqual(split["candidate_extraction"], "cpu")
                self.assertEqual(split["geometry_and_statistics"], "cpu")
                self.assertEqual(split["hybrid_steps"], {})

    def test_resident_cnr_export_is_reported_in_its_own_stage(self):
        runtime = _CountingRuntime({"vf_cnr_mask_u8_roi": 6})
        split = self._split([_Detector(routes={"cuda": 6})], resident=True, runtime=runtime)
        self.assertEqual(split["automatic_cnr_mask"], "device")
        self.assertEqual(
            split["hybrid_steps"], {"automatic_cnr_mask": ["vf_cnr_mask_u8_roi"]}
        )

    def test_a_runtime_without_metrics_does_not_break_result_assembly(self):
        class _BrokenRuntime(_Runtime):
            def performance_stats(self):
                raise RuntimeError("injected metrics failure")

        split = self._split([_Detector()], resident=True, runtime=_BrokenRuntime())
        self.assertEqual(split["candidate_extraction"], "cpu")
        self.assertEqual(split["geometry_and_statistics"], "cpu")

    def test_crossover_cpu_route_is_not_reported_as_device_preprocessing(self):
        detector = _Detector(gpu_active=True, routes={"cpu_crossover": 4}, crossover=True)
        split = self._split([detector], resident=True)
        self.assertEqual(split["preprocessing"], "cpu")
        self.assertEqual(split["candidate_extraction"], "cpu")

    def test_tiling_is_device_only_while_a_resident_image_exists(self):
        self.assertEqual(self._split([_Detector()], resident=True, tiling=True)["tiling_roi"], "device")
        self.assertEqual(self._split([_Detector()], resident=False, tiling=True)["tiling_roi"], "cpu")

    def test_split_is_present_in_the_public_execution_schema(self):
        source = (Path(__file__).resolve().parents[1] / "core" / "pipeline_stages.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"device_host_split": InspectionResultAssembler._device_host_split(', source)


if __name__ == "__main__":
    unittest.main()
