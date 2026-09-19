"""Template Anchor Grid localization: GPU routing, fallback, and equivalence with the CPU reference."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from core.gpu_runtime import GpuResidentImage, GpuRuntime
from core.tiler import GridAnchorConfig, Tiler


def _pattern_image(height: int = 400, width: int = 600) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(0, height, 40):
        for column in range(0, width, 40):
            image[row : row + 20, column : column + 20] = (row * 7 + column * 3) % 200 + 30
    image[150:200, 250:330] = (10, 220, 40)
    return image


class _FakeTemplateMatchDll:
    """Minimal DLL exposing the optional localization export and ABI probes."""

    def __init__(self, result=0, match=(250, 150, 80, 50), score=0.987):
        self.result = result
        self.match = match
        self.score = score
        self.calls = 0
        self.last_rect = None
        # ctypes-style export object, so the runtime can set argtypes/restype on it.
        self.vf_match_template_gray_u8 = _Stub(self._match)
        self.vf_match_template_gray_u8.argtypes = None

    def _match(
        self, _context, _generation, x, y, width, height,
        _templ, _templ_w, _templ_h, out_match, out_score,
    ):
        self.calls += 1
        self.last_rect = (x, y, width, height)
        if self.result != 0:
            return self.result
        out_match[0], out_match[1], out_match[2], out_match[3] = self.match
        out_score._obj.value = self.score
        return 0


class _NativeDll:
    """Stand-in for a DLL without the localization export."""


class _Stub:
    """Callable that accepts ctypes-style argtypes/restype assignment like a DLL export."""

    argtypes = None
    restype = None

    def __init__(self, callback=0):
        self.callback = callback

    def __call__(self, *args, **kwargs):
        return self.callback(*args, **kwargs) if callable(self.callback) else self.callback


def _with_plan_exports(dll):
    """The localization probe is gated behind the resident-ROI capability.

    A real DLL exports the plan ABI and creates a context; these stubs let the probe reach the
    new export and keep the routing test focused on the anchor backend.
    """
    def create_context(output):
        output._obj.value = 4242
        return 0

    dll.vf_context_create = create_context
    for name in (
        "vf_plan_query", "vf_plan_create", "vf_plan_execute", "vf_plan_destroy",
        "vf_dag_plan_query", "vf_dag_plan_create", "vf_dag_plan_execute", "vf_dag_plan_destroy",
        "vf_context_upload_u8", "vf_plan_execute_roi", "vf_dag_plan_execute_roi",
        "vf_context_destroy",
    ):
        if not hasattr(dll, name):
            setattr(dll, name, _Stub())
    return dll


def _runtime(dll) -> GpuRuntime:
    runtime = GpuRuntime(enabled=False, fallback_to_cpu=True)
    runtime._dll = _with_plan_exports(dll)
    runtime.device_count = 1
    runtime._load_optional_context()
    return runtime


def _resident(runtime, image) -> GpuResidentImage:
    """Register the image as resident without needing the full native upload ABI."""
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else int(image.shape[2])
    return GpuResidentImage(runtime, 1, width, height, channels)


class TilerAnchorBackendTests(unittest.TestCase):
    def _tiler(self, runtime, resident, **overrides):
        config = {
            "template_path": "template.png", "rows": 1, "cols": 1,
            "roi_w": 200, "roi_h": 120, "offset_x": 0, "offset_y": 0,
            "search_x": 0, "search_y": 0, "search_w": 0, "search_h": 0,
            "match_threshold": 0.0,
        }
        config.update(overrides)
        return Tiler(
            200, 120, anchor_config=GridAnchorConfig.from_dict(config),
            gpu_runtime=runtime, resident_image=resident,
            gpu_anchor_enabled=overrides.pop("gpu_anchor_enabled", True),
        )

    def _template_file(self, directory, image):
        path = directory / "template.png"
        cv2.imwrite(str(path), image[150:200, 250:330])
        return path

    def test_device_anchor_is_used_when_the_export_exists(self):
        import tempfile
        from pathlib import Path

        image = _pattern_image()
        with tempfile.TemporaryDirectory() as temporary:
            template_path = self._template_file(Path(temporary), image)
            dll = _FakeTemplateMatchDll()
            runtime = _runtime(dll)
            resident = _resident(runtime, image)
            tiler = self._tiler(runtime, resident, template_path=str(template_path))
            tiles = list(tiler.iter_tiles(image))

        self.assertEqual(dll.calls, 1)
        self.assertEqual(dll.last_rect, (0, 0, 600, 400))
        self.assertEqual(len(tiles), 1)
        metadata = tiles[0].metadata
        self.assertEqual(metadata["grid_anchor_backend"], "cuda_dll")
        self.assertEqual(metadata["match_bbox"], [250, 150, 80, 50])
        self.assertAlmostEqual(metadata["score"], 0.987, places=5)
        self.assertEqual((tiles[0].x, tiles[0].y), (250, 150))

    def test_unsupported_flat_template_falls_back_to_the_cpu_reference(self):
        import tempfile
        from pathlib import Path

        image = np.full((400, 600, 3), 77, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            template_path = self._template_file(Path(temporary), image)
            dll = _FakeTemplateMatchDll(result=8)   # VF_CUDA_UNSUPPORTED
            runtime = _runtime(dll)
            resident = _resident(runtime, image)
            tiler = self._tiler(runtime, resident, template_path=str(template_path))
            tiles = list(tiler.iter_tiles(image))

        self.assertEqual(dll.calls, 1)
        self.assertEqual(tiles[0].metadata["grid_anchor_backend"], "cpu")
        # The CPU reference picks the topmost-leftmost maximum, which is the origin here.
        self.assertEqual(tiles[0].metadata["match_bbox"], [0, 0, 80, 50])

    def test_missing_export_keeps_the_cpu_reference(self):
        import tempfile
        from pathlib import Path

        image = _pattern_image()
        with tempfile.TemporaryDirectory() as temporary:
            template_path = self._template_file(Path(temporary), image)
            runtime = _runtime(_NativeDll())
            resident = _resident(runtime, image)
            tiler = self._tiler(runtime, resident, template_path=str(template_path))
            tiles = list(tiler.iter_tiles(image))

        metadata = tiles[0].metadata
        self.assertEqual(metadata["grid_anchor_backend"], "cpu")
        self.assertEqual(metadata["match_bbox"], [250, 150, 80, 50])
        self.assertAlmostEqual(metadata["score"], 1.0, places=5)

    def test_resident_tiler_without_gpu_runtime_uses_the_resident_owner(self):
        """The pipeline builds a resident tiler with ``gpu_runtime=None`` so tiles are never
        re-cropped through CUDA. v1.6.0 then skipped device localization entirely; the anchor must
        reach the runtime that owns the resident image."""
        import tempfile
        from pathlib import Path

        image = _pattern_image()
        with tempfile.TemporaryDirectory() as temporary:
            template_path = self._template_file(Path(temporary), image)
            dll = _FakeTemplateMatchDll()
            runtime = _runtime(dll)
            resident = _resident(runtime, image)
            tiler = self._tiler(None, resident, template_path=str(template_path))
            tiles = list(tiler.iter_tiles(image))

        self.assertEqual(dll.calls, 1)
        self.assertEqual(tiles[0].metadata["grid_anchor_backend"], "cuda_dll")
        self.assertEqual(tiles[0].metadata["match_bbox"], [250, 150, 80, 50])
        self.assertIsNotNone(tiles[0].device_roi)

    def test_strict_cuda_surfaces_a_device_anchor_failure_instead_of_using_the_cpu(self):
        import tempfile
        from pathlib import Path

        from core.gpu_runtime import GpuRuntimeError

        image = _pattern_image()
        with tempfile.TemporaryDirectory() as temporary:
            template_path = self._template_file(Path(temporary), image)
            dll = _FakeTemplateMatchDll(result=1001)
            runtime = _runtime(dll)
            runtime.fallback_to_cpu = False
            resident = _resident(runtime, image)
            tiler = self._tiler(None, resident, template_path=str(template_path))
            with self.assertRaises(GpuRuntimeError):
                list(tiler.iter_tiles(image))

        self.assertEqual(dll.calls, 1)

    def test_recovered_anchor_failure_does_not_disable_later_gpu_steps(self):
        import tempfile
        from pathlib import Path

        image = _pattern_image()
        with tempfile.TemporaryDirectory() as temporary:
            template_path = self._template_file(Path(temporary), image)
            runtime = _runtime(_FakeTemplateMatchDll(result=1001))
            resident = _resident(runtime, image)
            tiler = self._tiler(None, resident, template_path=str(template_path))
            tiles = list(tiler.iter_tiles(image))

        self.assertEqual(tiles[0].metadata["grid_anchor_backend"], "cpu")
        self.assertEqual(runtime.last_error, "")

    def test_cpu_reference_converts_only_the_search_window_with_identical_results(self):
        """Gray conversion is per pixel, so a search-window conversion must pick the same anchor
        and score as matching inside a whole-image conversion."""
        import tempfile
        from pathlib import Path

        rng = np.random.default_rng(7)
        # A textured template (the pattern image's block at this spot is flat and would take the
        # SQDIFF branch instead of the TM_CCOEFF_NORMED reference compared below).
        image = rng.integers(0, 256, size=(400, 600, 3), dtype=np.uint8)
        search = (180, 90, 260, 180)
        with tempfile.TemporaryDirectory() as temporary:
            template_path = self._template_file(Path(temporary), image)
            template_gray = cv2.cvtColor(image[150:200, 250:330], cv2.COLOR_BGR2GRAY)
            tiler = self._tiler(
                None, None, template_path=str(template_path),
                search_x=search[0], search_y=search[1], search_w=search[2], search_h=search[3],
            )
            tiles = list(tiler.iter_tiles(image))

        full_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        window = full_gray[search[1]:search[1] + search[3], search[0]:search[0] + search[2]]
        scores = cv2.matchTemplate(window, template_gray, cv2.TM_CCOEFF_NORMED)
        _, best, _, location = cv2.minMaxLoc(scores)
        metadata = tiles[0].metadata
        self.assertEqual(metadata["grid_anchor_backend"], "cpu")
        self.assertEqual(
            metadata["match_bbox"], [search[0] + location[0], search[1] + location[1], 80, 50]
        )
        self.assertEqual(metadata["score"], float(best))

    def test_no_resident_image_keeps_the_cpu_reference(self):
        import tempfile
        from pathlib import Path

        image = _pattern_image()
        with tempfile.TemporaryDirectory() as temporary:
            template_path = self._template_file(Path(temporary), image)
            dll = _FakeTemplateMatchDll()
            runtime = _runtime(dll)
            tiler = self._tiler(runtime, None, template_path=str(template_path))
            tiles = list(tiler.iter_tiles(image))

        self.assertEqual(dll.calls, 0)
        self.assertEqual(tiles[0].metadata["grid_anchor_backend"], "cpu")
        self.assertEqual(tiles[0].metadata["match_bbox"], [250, 150, 80, 50])


if __name__ == "__main__":
    unittest.main()
