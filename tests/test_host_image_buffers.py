from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from core.gpu_runtime import GpuRuntime, GpuRuntimeError
from core.host_image_buffers import HostImageBufferPool
from core.image_loader import BmpReader
from tests.test_gpu_observability import _Function, _NativeDagPlanDll

ROOT = Path(__file__).resolve().parents[1]


class _PinningRuntime:
    """Minimal runtime double that records host registration calls."""

    def __init__(self, supported: bool = True, fail_register: bool = False):
        self.supports_host_register = supported
        self.fail_register = fail_register
        self.registered: list[int] = []
        self.unregistered: list[int] = []

    def register_host_buffer(self, buffer):
        if self.fail_register:
            raise GpuRuntimeError("register failed")
        self.registered.append(buffer.ctypes.data)

    def unregister_host_buffer(self, buffer):
        self.unregistered.append(buffer.ctypes.data)


def _write_bmp(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".bmp", image)
    if not ok:
        raise AssertionError("OpenCV could not encode the BMP fixture")
    encoded.tofile(str(path))


class HostImageBufferPoolTests(unittest.TestCase):
    def test_lease_reuses_one_backing_and_is_exclusive_per_run(self):
        pool = HostImageBufferPool(mode="pageable")
        with pool.lease() as first:
            backing = first((4, 12))
            address = backing.ctypes.data
            self.assertIsNone(first((4, 12)), "a run receives the pooled backing only once")
            with pool.lease() as concurrent:
                self.assertIsNone(concurrent((4, 12)), "a second concurrent lease must allocate normally")
            del backing
        self.assertEqual(first.details, {"reused": False, "pinned": False, "nbytes": 48})
        with pool.lease() as second:
            self.assertEqual(second((4, 12)).ctypes.data, address)
        self.assertEqual(second.details["reused"], True)
        self.assertEqual((pool.allocation_count, pool.reuse_count, pool.detached_count), (1, 1, 0))

    def test_surviving_view_detaches_the_backing_instead_of_overwriting_it(self):
        pool = HostImageBufferPool(mode="pageable")
        with pool.lease() as lease:
            backing = lease((2, 3))
            backing[:] = 7
            leaked_view = backing[:, :2]
            del backing
        with pool.lease() as lease:
            replacement = lease((2, 3))
            replacement[:] = 99
            del replacement
        self.assertEqual(pool.detached_count, 1)
        self.assertEqual(pool.allocation_count, 2)
        np.testing.assert_array_equal(leaked_view, np.full((2, 2), 7, dtype=np.uint8))

    def test_shape_change_off_mode_and_close(self):
        runtime = _PinningRuntime()
        pool = HostImageBufferPool(runtime, mode="auto")
        with pool.lease() as lease:
            first_address = lease((2, 6)).ctypes.data
        with pool.lease() as lease:
            self.assertEqual(lease((3, 6)).shape, (3, 6))
            self.assertTrue(lease.details["pinned"])
        self.assertEqual(runtime.unregistered, [first_address], "a replaced backing is unpinned first")
        pool.close()
        self.assertEqual(len(runtime.unregistered), 2)
        with pool.lease() as lease:
            self.assertIsNone(lease((3, 6)), "a closed pool never hands out a backing")

        off = HostImageBufferPool(runtime, mode="off")
        with off.lease() as lease:
            self.assertIsNone(lease((3, 6)))
        self.assertEqual(off.allocation_count, 0)

    def test_close_during_lease_unpins_after_release(self):
        runtime = _PinningRuntime()
        pool = HostImageBufferPool(runtime, mode="auto")
        lease = pool.lease()
        lease((2, 2))
        pool.close()
        self.assertEqual(runtime.unregistered, [], "the running inspection still owns the pinned pages")
        lease.close()
        self.assertEqual(len(runtime.unregistered), 1)

    def test_pinning_is_optional_and_failure_keeps_pageable_reuse(self):
        for runtime, expected in (
            (_PinningRuntime(supported=False), False),
            (_PinningRuntime(fail_register=True), False),
            (_PinningRuntime(), True),
        ):
            pool = HostImageBufferPool(runtime, mode="auto")
            with pool.lease() as lease:
                self.assertIsNotNone(lease((2, 2)))
            self.assertEqual(lease.details["pinned"], expected)
        pageable_runtime = _PinningRuntime()
        pool = HostImageBufferPool(pageable_runtime, mode="pageable")
        with pool.lease() as lease:
            lease((2, 2))
        self.assertEqual(pageable_runtime.registered, [])

    def test_environment_selects_mode_and_unknown_values_use_auto(self):
        with patch.dict("os.environ", {HostImageBufferPool.ENVIRONMENT_VARIABLE: "OFF"}):
            self.assertEqual(HostImageBufferPool().mode, "off")
        with patch.dict("os.environ", {HostImageBufferPool.ENVIRONMENT_VARIABLE: "fast"}):
            self.assertEqual(HostImageBufferPool().mode, "auto")


class BmpReaderBackingProviderTests(unittest.TestCase):
    def test_reused_backing_is_fully_overwritten_for_each_image(self):
        rng = np.random.default_rng(77)
        pool = HostImageBufferPool(mode="pageable")
        reader = BmpReader(max_workers=2)
        with tempfile.TemporaryDirectory() as directory:
            for index in range(3):
                # Odd width exercises the BMP row padding bytes inside the reused backing.
                image = rng.integers(0, 256, size=(33, 17, 3), dtype=np.uint8)
                path = Path(directory) / f"image_{index}.bmp"
                _write_bmp(path, image)
                with pool.lease() as lease:
                    decoded = reader.read(path, preserve_file_order=True, backing_provider=lease)
                    np.testing.assert_array_equal(decoded, image)
                    np.testing.assert_array_equal(decoded, cv2.imread(str(path), cv2.IMREAD_COLOR))
                    del decoded
                if index == 0:
                    with pool.lease() as lease:
                        stale = lease((33, ((17 * 3 + 3) // 4) * 4))
                        stale[:] = 255
                        del stale
        self.assertEqual(pool.allocation_count, 1)
        self.assertEqual(pool.detached_count, 0)

    def test_unusable_backing_is_ignored(self):
        image = np.arange(5 * 4 * 3, dtype=np.uint8).reshape(5, 4, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.bmp"
            _write_bmp(path, image)
            stride = ((4 * 3 + 3) // 4) * 4
            for bad in (
                np.zeros((5, stride + 1), dtype=np.uint8),
                np.zeros((5, stride), dtype=np.uint16),
                np.zeros((5, stride * 2), dtype=np.uint8)[:, ::2],
                None,
            ):
                decoded = BmpReader(max_workers=1).read(
                    path, preserve_file_order=True, backing_provider=lambda _shape, bad=bad: bad
                )
                np.testing.assert_array_equal(decoded, image)
            # Non file-order reads never ask for a backing.
            provider = Mock(return_value=None)
            BmpReader(max_workers=1).read(path, backing_provider=provider)
            provider.assert_not_called()


class GpuRuntimeHostRegisterTests(unittest.TestCase):
    def _runtime(self, with_exports: bool):
        runtime = GpuRuntime(enabled=False)
        dll = _NativeDagPlanDll()
        dll.registered, dll.unregistered = [], []
        if with_exports:
            dll.vf_host_register_u8 = _Function(lambda _c, ptr, size: dll.registered.append(int(size)) or 0)
            dll.vf_host_unregister_u8 = _Function(lambda _c, _ptr: dll.unregistered.append(1) or 0)
        runtime._dll = dll
        runtime.device_count = 1
        runtime._load_optional_context()
        return runtime, dll

    def test_register_bridge_and_old_dll_capability(self):
        runtime, dll = self._runtime(with_exports=True)
        self.assertTrue(runtime.supports_host_register)
        self.assertTrue(runtime.status(True)["capabilities"]["host_register"])
        buffer = np.zeros((3, 8), dtype=np.uint8)
        runtime.register_host_buffer(buffer)
        runtime.unregister_host_buffer(buffer)
        self.assertEqual((dll.registered, dll.unregistered), ([24], [1]))
        with self.assertRaises(GpuRuntimeError):
            runtime.register_host_buffer(np.zeros((3, 8), dtype=np.uint16))

        old_runtime, _old_dll = self._runtime(with_exports=False)
        self.assertFalse(old_runtime.supports_host_register)
        with self.assertRaises(GpuRuntimeError):
            old_runtime.register_host_buffer(buffer)

    def test_native_register_failure_raises_without_marking_the_run_failed(self):
        runtime, dll = self._runtime(with_exports=True)
        dll.vf_host_register_u8 = _Function(lambda *_args: 3)
        with self.assertRaises(GpuRuntimeError):
            runtime.register_host_buffer(np.zeros((2, 2), dtype=np.uint8))
        self.assertEqual(runtime.last_error, "")


class PipelineHostImageBufferTests(unittest.TestCase):
    def test_sequential_bmp_runs_reuse_pinned_backing_and_upload_each_image_exactly(self):
        from core.gpu_session import GpuExecutionSession
        from core.pipeline import AOIPipeline
        from detectors.detector_401 import Detector401

        runtime = GpuRuntime(enabled=False, fallback_to_cpu=True)
        dll = _NativeDagPlanDll()
        dll.registered, dll.unregistered = [], []
        dll.vf_host_register_u8 = _Function(lambda _c, _ptr, size: dll.registered.append(int(size)) or 0)
        dll.vf_host_unregister_u8 = _Function(lambda _c, _ptr: dll.unregistered.append(1) or 0)
        runtime._dll = dll
        runtime.device_count = 1
        runtime._load_optional_context()

        recipe_path = ROOT / "recipes" / "PRODUCT_A_NEGATIVE_401_AOI_01.yaml"
        recipe = deepcopy(AOIPipeline(recipe_path, ROOT / "outputs").recipe_manager.load(recipe_path))
        recipe["gpu"] = {"mode": "auto", "dll_path": "fake_resident.dll", "fallback_to_cpu": True, "tiling": False}
        recipe["tile"] = {**recipe["tile"], "mode": "grid"}
        recipe["detectors"]["401-AS-SN-1"]["use_gpu"] = True
        session = GpuExecutionSession(runtime, requested=True, config=recipe["gpu"])
        session.host_image_buffers = HostImageBufferPool(runtime, mode="auto")
        overrides = {key: False for key in ("save_overlay", "save_ng_tiles", "save_csv", "save_matrix_csv", "save_json")}
        params = {
            "roi_inset_px": 0, "blur_size": 3, "morph_operation": "open", "morph_kernel": 3,
            "morph_iterations": 1, "adaptive_block_size": 3, "adaptive_c": 2.0, "min_area": 0, "max_area": 0,
        }
        rng = np.random.default_rng(401)
        reports = []
        with tempfile.TemporaryDirectory(prefix="visionflow_host_buffer_") as temporary:
            root = Path(temporary)
            for index in range(3):
                image = rng.integers(0, 256, size=(96, 130, 3), dtype=np.uint8)
                image_path = root / f"input_{index}.bmp"
                _write_bmp(image_path, image)
                pipeline = AOIPipeline(recipe_path, root, output_overrides=overrides, gpu_session=session)
                pipeline.recipe_manager.load = Mock(return_value=recipe)
                pipeline.detector_manager.create_enabled = Mock(
                    return_value=[Detector401(params=params, use_gpu=True, gpu_runtime=runtime)]
                )
                reports.append(pipeline.run(image_path))
                np.testing.assert_array_equal(dll.resident, image, err_msg=f"image {index} upload")

        buffers = [report["execution"]["gpu"]["resident_image"]["host_buffer"] for report in reports]
        self.assertEqual([item["reused"] for item in buffers], [False, True, True])
        self.assertTrue(all(item["pinned"] for item in buffers))
        pool = session.host_image_buffers
        self.assertEqual((pool.allocation_count, pool.reuse_count, pool.detached_count), (1, 2, 0))
        self.assertEqual(len(dll.registered), 1)
        session.close()
        self.assertEqual(len(dll.unregistered), 1, "closing the session unpins before the runtime closes")


if __name__ == "__main__":
    unittest.main()
