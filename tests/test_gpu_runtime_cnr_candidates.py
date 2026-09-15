"""GpuRuntime binding for the resident 202 candidate export: layout, retry and error reporting."""

from __future__ import annotations

import unittest
from unittest.mock import PropertyMock, patch

import numpy as np

from core.gpu_runtime import GpuResidentImage, GpuRuntime, GpuRuntimeError


class _Stub:
    argtypes = None
    restype = None

    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


class _CandidateDll:
    """Fake export: reports `available` candidates and needs a capacity of at least that many."""

    def __init__(self, available=3, status=0, result=0):
        self.available = available
        self.status = status
        self.result = result
        self.calls = []
        self.vf_cnr_candidates_u8_roi = _Stub(self._candidates)

    def _candidates(self, _context, generation, x, y, width, height, ints, int_count, reals, real_count,
                    median, mad, threshold, records, stats, capacity, count, components, status):
        self.calls.append({
            "generation": generation.value, "roi": (x, y, width, height), "int_count": int_count,
            "real_count": real_count, "capacity": capacity, "first_int": ints[0], "last_real": reals[5],
        })
        median._obj.value = 1.5
        mad._obj.value = 0.25
        threshold._obj.value = 8.0
        count._obj.value = self.available
        components._obj.value = self.available + 2
        if self.result:
            status._obj.value = self.status
            return self.result
        if capacity < self.available:
            status._obj.value = 2
            return 8
        for index in range(self.available):
            for field in range(7):
                records[index * 7 + field] = index * 10 + field
            for field in range(3):
                stats[index * 3 + field] = index + field / 4.0
        status._obj.value = 0
        return 0


class GpuRuntimeCnrCandidateTests(unittest.TestCase):
    def _runtime(self, dll):
        runtime = GpuRuntime(enabled=False, fallback_to_cpu=True)
        runtime._dll = dll
        runtime._context = 1
        runtime._error_message = lambda code: f"code {code}"
        return runtime

    def _call(self, runtime, capacity=16):
        roi = GpuResidentImage(runtime, 7, 64, 48, 3).roi(4, 2, 40, 30)
        with patch.object(GpuRuntime, "supports_cnr_candidates_u8_roi", new_callable=PropertyMock, return_value=True), \
                patch.object(GpuRuntime, "_require_gaussian_f32_sigma", return_value=None):
            return runtime.cnr_candidates_u8_roi(
                roi, list(range(51, 73)), [0.0, 3.0, 8.0, 1e-6, 1.4826, 1.5], candidate_capacity=capacity
            )

    def test_records_and_scalars_are_returned_in_the_documented_layout(self):
        dll = _CandidateDll(available=3)
        result = self._call(self._runtime(dll))
        call = dll.calls[0]
        self.assertEqual((call["generation"], call["roi"]), (7, (4, 2, 40, 30)))
        self.assertEqual((call["int_count"], call["real_count"], call["first_int"], call["last_real"]), (22, 6, 51, 1.5))
        self.assertEqual(result["records"].shape, (3, 7))
        self.assertEqual(result["stats"].shape, (3, 3))
        np.testing.assert_array_equal(result["records"][2], [20, 21, 22, 23, 24, 25, 26])
        self.assertEqual(result["stats"].dtype, np.float32)
        self.assertEqual(result["component_count"], 5)
        self.assertEqual(float(result["residual_median"]), 1.5)
        self.assertEqual(result["threshold"], 8.0)

    def test_a_small_record_buffer_is_retried_once_with_the_reported_count(self):
        dll = _CandidateDll(available=40)
        result = self._call(self._runtime(dll), capacity=4)
        self.assertEqual([call["capacity"] for call in dll.calls], [4, 40])
        self.assertEqual(result["records"].shape, (40, 7))

    def test_unsupported_statuses_raise_with_the_status_attached(self):
        dll = _CandidateDll(available=2, status=1, result=8)
        with self.assertRaises(GpuRuntimeError) as raised:
            self._call(self._runtime(dll))
        self.assertEqual(raised.exception.error_code, 8)
        self.assertEqual(raised.exception.candidate_status, 1)
        self.assertIn("min_background_pixels", str(raised.exception))
        self.assertEqual(len(dll.calls), 1)

    def test_wrong_parameter_counts_are_rejected_before_calling_the_dll(self):
        dll = _CandidateDll()
        runtime = self._runtime(dll)
        roi = GpuResidentImage(runtime, 7, 64, 48, 3).roi(0, 0, 10, 10)
        with patch.object(GpuRuntime, "supports_cnr_candidates_u8_roi", new_callable=PropertyMock, return_value=True):
            with self.assertRaises(GpuRuntimeError):
                runtime.cnr_candidates_u8_roi(roi, [1] * 21, [0.0] * 6)
        self.assertEqual(dll.calls, [])


if __name__ == "__main__":
    unittest.main()
