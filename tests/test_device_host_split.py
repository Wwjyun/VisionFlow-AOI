"""The result must report the actual device/host split of each pipeline step.

AGENT.md requires reporting what actually ran where instead of describing the flow as fully GPU,
so these tests pin the reported split for the CPU path, the resident-device path, and the case
where the crossover measured the CPU route.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from core.gpu_metrics import performance_stats_delta
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
    def _split(self, detectors, resident=True, tiling=False, runtime=None, anchor_on_device=False):
        return InspectionResultAssembler._device_host_split(
            gpu_runtime=runtime if runtime is not None else _Runtime(),
            detectors=detectors,
            resident_image=object() if resident else None,
            tiling_gpu_requested=tiling,
            anchor_on_device=anchor_on_device,
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
        split = self._split([detector], resident=True, anchor_on_device=True)
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

    def test_second_run_does_not_reuse_first_run_export_counts(self):
        runtime = _CountingRuntime({"vf_median_f32": 12})
        baseline = runtime.performance_stats()

        split = InspectionResultAssembler._device_host_split(
            gpu_runtime=runtime,
            detectors=[_Detector(routes={"cpu": 1})],
            resident_image=object(),
            tiling_gpu_requested=False,
            gpu_metrics_baseline=baseline,
        )

        self.assertEqual(split["candidate_extraction"], "cpu")
        self.assertEqual(split["hybrid_steps"], {})

    def test_current_run_metrics_are_deltas_and_cumulative_metrics_are_preserved(self):
        from core.performance import PipelineProfiler

        class _Manager:
            @staticmethod
            def ai_performance_stats():
                return {}

        runtime = _CountingRuntime({"vf_median_f32": 12})
        baseline = runtime.performance_stats()
        runtime._functions["vf_median_f32"]["calls"] = 15
        result = InspectionResultAssembler.build(
            image_path=Path("input.png"), started=0.0,
            recipe={"recipe_name": "r", "machine_id": "m", "product_id": "p", "version": "1"},
            provenance={}, aggregate={"final_result": "PASS", "summary": {}},
            tile_results=[], detector_manager=_Manager(), detectors=[_Detector()],
            gpu_runtime=runtime, gpu_mode="auto", tiling_gpu_requested=False,
            display_requested=False, resident_image=None, profiler=PipelineProfiler(),
            gpu_metrics_baseline=baseline,
        )["execution"]["gpu"]

        self.assertEqual(result["metrics"]["functions"]["vf_median_f32"]["calls"], 3)
        self.assertEqual(result["metrics_cumulative"]["functions"]["vf_median_f32"]["calls"], 15)
        self.assertEqual(result["metrics"]["measurement_period"], "current_pipeline_run")

    def test_metrics_delta_covers_transfers_timings_and_drops_stale_native_timing(self):
        baseline = {
            "load_sec": 0.25, "call_count": 4, "estimated_round_trips": 4,
            "host_to_device_bytes": 100, "device_to_host_bytes": 40,
            "wall_sec": 0.5, "lock_wait_sec": 0.1, "kernel_launch_count": 8,
            "native_cumulative_ms": {"kernel_ms": 3.0},
            "native_timings_ms": {"kernel_ms": 1.0},
            "functions": {"vf_median_f32": {
                "calls": 4, "host_to_device_bytes": 100, "device_to_host_bytes": 40,
                "wall_sec": 0.5, "lock_wait_sec": 0.1,
            }},
        }
        current = {**baseline, "native_timings_ms": {"kernel_ms": 99.0}}

        no_calls = performance_stats_delta(current, baseline)

        self.assertEqual(no_calls["load_sec"], 0.0)
        self.assertEqual(no_calls["call_count"], 0)
        self.assertEqual(no_calls["functions"], {})
        self.assertIsNone(no_calls["native_timings_ms"])

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

    def test_resident_upload_alone_does_not_report_device_anchor_localization(self):
        """v1.6.0 reported ``device`` whenever the image was resident, even though the anchor ran
        on the CPU reference. Only the backend recorded on the tiles may claim device work."""
        split = self._split([_Detector(routes={"cuda": 6})], resident=True, anchor_on_device=False)
        self.assertEqual(split["resident_upload"], "device")
        self.assertEqual(split["anchor_localization"], "cpu")

    def test_assembled_result_reads_the_anchor_backend_from_tile_metadata(self):
        from core.performance import PipelineProfiler

        class _Manager:
            @staticmethod
            def ai_performance_stats():
                return {}

        class _StatusRuntime(_Runtime):
            @staticmethod
            def performance_stats():
                return {"functions": {}}

        def assemble(backend):
            tile = {"tile": {"tile_id": "r0000_c0000", "metadata": {"grid_anchor_backend": backend}}}
            return InspectionResultAssembler.build(
                image_path=Path("input.png"), started=0.0,
                recipe={"recipe_name": "r", "machine_id": "m", "product_id": "p", "version": "1"},
                provenance={}, aggregate={"final_result": "PASS", "summary": {}},
                tile_results=[tile], detector_manager=_Manager(), detectors=[],
                gpu_runtime=_StatusRuntime(), gpu_mode="auto", tiling_gpu_requested=False,
                display_requested=False, resident_image=None, profiler=PipelineProfiler(),
            )["execution"]["gpu"]["device_host_split"]["anchor_localization"]

        self.assertEqual(assemble("cuda_dll"), "device")
        self.assertEqual(assemble("cpu"), "cpu")

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
