"""Widened final-output equivalence matrix for the 202-CS-SN-1 detector with the device float32
Gaussian background.

For every scene the detector is run twice with identical parameters and the same CUDA runtime (so
the exact median is on the device in both runs); the only difference is which implementation of
``cv2.GaussianBlur(float32, (k, k), 0.0)`` builds the background:

  reference : ``cv2.GaussianBlur`` (the committed CPU path)
  device    : ``GpuRuntime.gaussian_blur_f32`` (vf_gaussian_blur_f32), installed for that one call
              site by replacing the detector module's ``cv2`` name with a pass-through proxy.

The comparison is strict. PASS/NG, defect count and every defect field (bbox_local, area,
confidence, type and the whole metadata mapping) are compared with ``==``; any mismatch is reported
field by field with both values. The candidate mask of the background step is additionally compared
byte for byte, because it is what makes bbox/area/count stable.

Run with the repository Python:

    .\\env\\Scripts\\python.exe tools\\gaussian_202_matrix.py
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gpu_runtime import GpuRuntime  # noqa: E402
from detectors.detector_202_1 import Detector202_1  # noqa: E402

EVIDENCE_DIR = ROOT / "outputs_validation" / "cnr_profile"
EVIDENCE_TEXT = EVIDENCE_DIR / "gaussian_202_final_output_matrix.txt"
EVIDENCE_JSON = EVIDENCE_DIR / "gaussian_202_final_output_matrix.json"

# Fields whose value is derived from the residual (image - background) and therefore moves by the
# background tolerance itself. They are compared strictly AND summarized separately so a reader can
# see that nothing structural moved.
STRUCTURAL_DEFECT_FIELDS = ("type", "bbox_local", "area", "confidence")
RESIDUAL_METADATA_FIELDS = (
    "cnr", "contrast", "defect_mean", "background_mean", "background_std", "background_area_px",
    "residual_median", "mad", "robust_noise_sigma", "residual_threshold",
)
STRUCTURAL_METADATA_FIELDS = (
    "method", "background_kernel", "background_kernel_config", "gaussian_sigma", "mad_scale",
    "noise_sigma_floor", "residual_threshold_floor", "residual_sigma_multiplier",
    "candidate_max_value", "morphology", "connectivity", "min_area", "max_area",
)


class _SubstitutionProxy:
    """Pass-through facade over cv2 whose GaussianBlur is the device export."""

    def __init__(self, real_cv2, runtime):
        self._cv2 = real_cv2
        self._runtime = runtime
        self.calls: list[tuple[str, tuple, float]] = []

    def __getattr__(self, name):
        return getattr(self._cv2, name)

    def GaussianBlur(self, src, ksize, sigma1=0.0, *args, **kwargs):  # noqa: N802 - cv2 name
        ksize = tuple(int(value) for value in ksize) if isinstance(ksize, (tuple, list)) else ksize
        # sigma1 is forwarded as the double the detector's gaussian_sigma parameter carries; the
        # native export applies OpenCV's own rule to it, so a non-zero sigma is never substituted.
        usable = (
            isinstance(ksize, tuple)
            and len(ksize) == 2
            and ksize[0] == ksize[1]
            and not args
            and not kwargs
            and np.asarray(src).dtype == np.float32
        )
        if not usable:
            self.calls.append(("cpu", ksize if isinstance(ksize, tuple) else (ksize, ksize), float(sigma1)))
            return self._cv2.GaussianBlur(src, ksize, sigma1, *args, **kwargs)
        self.calls.append(("device", ksize, float(sigma1)))
        return self._runtime.gaussian_blur_f32(src, int(ksize[0]), float(sigma1))


@contextlib.contextmanager
def device_gaussian(module, runtime):
    """Replace one detector module's cv2 name for the duration of the block."""
    proxy = _SubstitutionProxy(cv2, runtime)
    original = module.cv2
    module.cv2 = proxy
    try:
        yield proxy
    finally:
        module.cv2 = original


def blob(gray: np.ndarray, x: int, y: int, width: int, height: int, delta: float) -> None:
    """Add one rectangular defect with the given contrast step, clipped to the uint8 domain."""
    region = gray[y:y + height, x:x + width].astype(np.float32) + float(delta)
    gray[y:y + height, x:x + width] = np.clip(region, 0, 255).astype(np.uint8)


def sweep_grid(width: int, height: int, deltas, sizes, spacing: int = 34) -> np.ndarray:
    """A lattice of defects that spans small areas and low contrasts at once."""
    gray = np.full((height, width), 150, dtype=np.uint8)
    for row, delta in enumerate(deltas):
        for column, size in enumerate(sizes):
            x = 20 + column * spacing
            y = 20 + row * spacing
            if x + size >= width - 2 or y + size >= height - 2:
                continue
            blob(gray, x, y, size, size, delta)
    return gray


def scene_gray(scene: dict) -> np.ndarray:
    height, width = scene["shape"]
    rng = np.random.default_rng(scene["seed"])
    if scene.get("sweep"):
        gray = sweep_grid(width, height, scene["deltas"], scene["sizes"])
    else:
        gray = np.full((height, width), scene.get("base", 150), dtype=np.uint8)
        for (x, y, dw, dh, delta) in scene.get("defects", ()):
            blob(gray, x, y, dw, dh, delta)
    noise = float(scene.get("noise", 0.0))
    if noise > 0.0:
        noisy = gray.astype(np.float32) + rng.normal(0.0, noise, gray.shape)
        gray = np.clip(noisy, 0, 255).astype(np.uint8)
    return gray


def build_scenes() -> list[dict]:
    """Scenes with varied defect size/contrast/count plus engineered acceptance boundaries.

    The detector subtracts a Gaussian background whose sigma grows with the kernel (sigma = 5.0 for
    the automatic kernel 31 of a 400x600 image), so a defect only leaves a large residual when its
    size is comparable to the kernel support: the contrast ladder below therefore uses 30 px
    defects, and the small-defect cases use high contrast or the 51/127 kernels.
    """
    small = (400, 600)
    panel = (800, 1200)
    ladder = [(60 + 70 * index, 200, 30, 30, delta)
              for index, delta in enumerate((6, 7, 8, 9, 10, 12, 15, 20))]
    # The mask of a solid defect is a thin ring around its edge (the blurred background equals the
    # defect value inside it), and that ring only survives the 3x3 opening above a sharp size cliff.
    # These sizes straddle the measured cliff for the automatic kernel 31 and the explicit 51.
    kernel31_cliff = [(100 + 240 * index, 100, size, size, 150)
                      for index, size in enumerate((182, 184, 186, 188, 190))]
    kernel51_cliff = [(100 + 180 * index, 100, size, size, 150)
                      for index, size in enumerate((172, 174, 176, 178, 180))]
    scenes: list[dict] = [
        # --- the size cliff of the automatic kernel 31 --------------------------------
        {"name": "size-below-cliff-180", "shape": small, "seed": 1,
         "defects": [(200, 100, 180, 180, 150)]},
        {"name": "size-at-cliff-186", "shape": small, "seed": 2,
         "defects": [(200, 100, 186, 186, 150)]},
        {"name": "size-above-cliff-190", "shape": small, "seed": 3,
         "defects": [(200, 100, 190, 190, 150)]},
        {"name": "size-above-cliff-210", "shape": small, "seed": 4,
         "defects": [(200, 100, 210, 210, 150)]},
        {"name": "size-cliff-182-190", "shape": (1000, 1000), "seed": 5,
         "defects": kernel31_cliff},
        {"name": "size-cliff-172-180-kernel51", "shape": (1000, 1000), "seed": 6,
         "defects": kernel51_cliff, "params": {"background_kernel_size": 51}},
        {"name": "size-cliff-dark-defect", "shape": (1000, 1000), "seed": 7,
         "defects": [(100 + 240 * index, 500, size, size, -140)
                     for index, size in enumerate((184, 186, 188, 190))]},
        # --- single defects over the detected range -----------------------------------
        {"name": "single-40px-delta40", "shape": small, "seed": 8,
         "defects": [(200, 100, 40, 40, 40)]},
        {"name": "single-cluster-120px-delta80", "shape": small, "seed": 9,
         "defects": [(240, 140, 120, 120, 80)]},
        {"name": "single-dark-40px-delta-60", "shape": small, "seed": 10,
         "defects": [(260, 160, 40, 40, -60)]},
        {"name": "single-line-40x120", "shape": small, "seed": 11,
         "defects": [(280, 140, 40, 120, 150)]},
        # --- contrast ladder: eight 30 px defects straddling the residual threshold --
        {"name": "contrast-ladder-6-to-20", "shape": small, "seed": 12, "defects": ladder},
        # --- counts -----------------------------------------------------------------
        {"name": "two-defects-mixed", "shape": small, "seed": 13,
         "defects": [(100, 100, 30, 30, 60), (420, 320, 25, 25, 25)]},
        {"name": "five-defects-mixed", "shape": small, "seed": 14,
         "defects": [(60, 60, 20, 20, 120), (200, 90, 30, 30, 45), (330, 150, 40, 40, 30),
                     (120, 300, 25, 60, 70), (400, 340, 60, 25, 20)]},
        {"name": "twelve-defects-mixed", "shape": small, "seed": 15,
         "defects": [(40 + 90 * (index % 5), 70 + 120 * (index // 5), 20 + (index % 3) * 10,
                      20 + (index % 4) * 10, 15 + (index % 6) * 12) for index in range(12)]},
        # --- noise ------------------------------------------------------------------
        {"name": "noise1-single-30px", "shape": small, "seed": 16, "noise": 1.0,
         "defects": [(280, 180, 30, 30, 30)]},
        {"name": "noise2-multi", "shape": small, "seed": 17, "noise": 2.0,
         "defects": [(280, 180, 30, 30, 25), (150, 300, 40, 30, 18)]},
        {"name": "noise4-clean-run", "shape": small, "seed": 18, "noise": 4.0, "defects": []},
        {"name": "noise6-weak-defect", "shape": small, "seed": 19, "noise": 6.0,
         "defects": [(280, 180, 40, 40, 30)]},
        {"name": "noise3-multi-weak", "shape": small, "seed": 20, "noise": 3.0,
         "defects": [(280, 180, 30, 30, 14), (380, 300, 35, 35, 12), (120, 120, 60, 20, 11)]},
        # --- engineered acceptance boundaries ---------------------------------------
        # residual threshold floor 8.0 on a noise-free background, peak at the boundary
        {"name": "cnr-boundary-delta9", "shape": small, "seed": 21,
         "defects": [(280, 180, 20, 20, 9)]},
        {"name": "cnr-boundary-delta11", "shape": small, "seed": 22,
         "defects": [(280, 180, 20, 20, 11)]},
        # threshold driven by 3 * robust sigma instead of the floor (noise 3, floor disabled)
        {"name": "cnr-sigma-boundary", "shape": small, "seed": 23, "noise": 3.0,
         "defects": [(280, 180, 30, 30, 14)], "params": {"residual_threshold_floor": 0.0}},
        # component area exactly at the limit: a 190 px defect yields two 370 px components, so
        # max=370 accepts (the filter rejects only area > max) and the 200 px defect is rejected.
        {"name": "area-max-exact-370", "shape": panel, "seed": 24,
         "defects": [(100, 200, 190, 190, 150), (500, 200, 200, 200, 150)],
         "params": {"max_component_area_px": 370}},
        {"name": "area-max-just-below-370", "shape": panel, "seed": 25,
         "defects": [(100, 200, 190, 190, 150), (500, 200, 200, 200, 150)],
         "params": {"max_component_area_px": 369}},
        {"name": "area-min-exact-370", "shape": panel, "seed": 26,
         "defects": [(100, 200, 190, 190, 150), (500, 200, 200, 200, 150)],
         "params": {"min_component_area_px": 370}},
        {"name": "area-min-just-above-370", "shape": panel, "seed": 27,
         "defects": [(100, 200, 190, 190, 150), (500, 200, 200, 200, 150)],
         "params": {"min_component_area_px": 371}},
        # component area boundary from below with morphology disabled on a small defect
        {"name": "area-min-boundary-morph-off", "shape": small, "seed": 28,
         "defects": [(120, 120, 40, 40, 150), (300, 120, 120, 120, 150)],
         "params": {"morph_operation": "none", "min_component_area_px": 2000}},
        # --- kernels and shapes ------------------------------------------------------
        {"name": "kernel31-explicit", "shape": small, "seed": 29,
         "defects": [(200, 100, 190, 190, 150)], "params": {"background_kernel_size": 31}},
        {"name": "kernel51-explicit", "shape": small, "seed": 30,
         "defects": [(200, 100, 190, 190, 150)], "params": {"background_kernel_size": 51}},
        {"name": "kernel127-explicit", "shape": small, "seed": 31,
         "defects": [(200, 100, 120, 120, 150)], "params": {"background_kernel_size": 127}},
        {"name": "kernel51-dark-background", "shape": small, "seed": 32, "base": 40,
         "defects": [(200, 100, 190, 190, 150)], "params": {"background_kernel_size": 51}},
        {"name": "wide-strip-1024x192", "shape": (192, 1024), "seed": 33,
         "defects": [(300, 80, 120, 120, 150)], "params": {"background_kernel_size": 51}},
        {"name": "tall-strip-1920x256", "shape": (1920, 256), "seed": 34,
         "defects": [(120, 960, 120, 120, 150)], "params": {"background_kernel_size": 51}},
        # production auto kernel: min(h, w) // 40 = 51 for the profiled 4000x2000 ROI
        {"name": "auto-kernel51-2048x2048", "shape": (2048, 2048), "seed": 35,
         "defects": [(1000, 1024, 120, 120, 150), (500, 500, 190, 190, 150)]},
        # spanning sweep: lattice over sizes and contrasts (many boundary candidates per scene)
        {"name": "sweep-grid-no-noise", "shape": (260, 260), "seed": 36, "sweep": True,
         "deltas": (12, 30, 60, 120, 200, 250), "sizes": (3, 4, 5, 7, 11, 15)},
        {"name": "sweep-grid-noise2", "shape": (260, 260), "seed": 37, "sweep": True, "noise": 2.0,
         "deltas": (12, 25, 45, 80, 150, 250), "sizes": (3, 4, 5, 8, 13, 21)},
        {"name": "sweep-grid-kernel51", "shape": (300, 300), "seed": 38, "sweep": True,
         "deltas": (15, 25, 40, 70, 120, 200), "sizes": (5, 8, 13, 21, 34, 48),
         "params": {"background_kernel_size": 51}},
        {"name": "sweep-grid-no-noise-kernel127", "shape": (300, 300), "seed": 39, "sweep": True,
         "deltas": (15, 30, 60, 120, 200, 250), "sizes": (5, 13, 21, 34, 60, 90),
         "params": {"background_kernel_size": 127}},
        # --- explicit gaussian_sigma (an admin recipe parameter of the detector) --------------
        # With sigma != 0 the device must apply that sigma, not OpenCV's automatic rule; before the
        # sigma fix these scenes produced a different candidate mask.
        {"name": "sigma1-no-noise", "shape": small, "seed": 40,
         "defects": [(200, 100, 190, 190, 150)], "params": {"gaussian_sigma": 1.0}},
        {"name": "sigma1-multi", "shape": small, "seed": 41,
         "defects": [(60, 60, 20, 20, 120), (200, 90, 30, 30, 45), (330, 150, 40, 40, 30),
                     (120, 300, 25, 60, 70)],
         "params": {"gaussian_sigma": 1.0}},
        {"name": "sigma2.5-kernel31", "shape": small, "seed": 42,
         "defects": [(200, 100, 40, 40, 40)], "params": {"gaussian_sigma": 2.5}},
        {"name": "sigma5-kernel31", "shape": small, "seed": 43,
         "defects": [(200, 100, 60, 60, 60)], "params": {"gaussian_sigma": 5.0}},
        {"name": "sigma1.25-kernel51", "shape": small, "seed": 44,
         "defects": [(200, 100, 60, 60, 60)],
         "params": {"background_kernel_size": 51, "gaussian_sigma": 1.25}},
        {"name": "sigma0.5-noise2-sweep", "shape": (300, 300), "seed": 45, "noise": 2.0,
         "sweep": True, "deltas": (15, 25, 45, 80, 150, 250), "sizes": (5, 8, 13, 21, 34, 48),
         "params": {"background_kernel_size": 31, "gaussian_sigma": 0.5}},
        {"name": "sigma2-kernel127-large", "shape": (1000, 1000), "seed": 46,
         "defects": kernel31_cliff, "params": {"background_kernel_size": 127, "gaussian_sigma": 2.0}},
        {"name": "sigma3-dark-background", "shape": small, "seed": 47, "base": 40,
         "defects": [(200, 100, 120, 120, 150)],
         "params": {"background_kernel_size": 51, "gaussian_sigma": 3.0}},
    ]
    return scenes


def run_case(scene: dict, runtime: GpuRuntime, module) -> dict:
    gray = scene_gray(scene)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    params = dict(scene.get("params", ()))

    reference_detector = Detector202_1(params=params, use_gpu=True, gpu_runtime=runtime)
    device_detector = Detector202_1(params=params, use_gpu=True, gpu_runtime=runtime)

    # Count exact-median calls so the evidence shows the residual median stayed on the device in
    # BOTH modes; otherwise a CPU-median fallback could hide a difference in the background step.
    median_calls = {"reference": 0, "device": 0}
    native_median = runtime.median_f32

    def counted(bucket):
        def call(values):
            median_calls[bucket] += 1
            return native_median(values)
        return call

    try:
        runtime.median_f32 = counted("reference")
        reference = reference_detector.run(image)
        runtime.median_f32 = counted("device")
        with device_gaussian(module, runtime) as proxy:
            device = device_detector.run(image)
            blur_calls = list(proxy.calls)
    finally:
        runtime.median_f32 = native_median

    # The mask of the background step on its own: bit equality here is what keeps every component
    # (and therefore every bbox, area and count) stable.
    reference_mask = reference_detector._automatic_cnr_mask(reference_detector._make_gray(image))["candidate_mask"]
    with device_gaussian(module, runtime):
        device_mask = device_detector._automatic_cnr_mask(device_detector._make_gray(image))["candidate_mask"]
    mask_equal = bool(np.array_equal(reference_mask, device_mask))

    differences = compare_results(reference, device)
    residual_deltas = summarize_residual_deltas(reference, device)
    # [compared, equal] per field key, so a field with several defects is counted per defect.
    field_equality: dict[str, list[int]] = {}
    for key, left, right in compared_fields(reference, device):
        totals = field_equality.setdefault(key, [0, 0])
        totals[0] += 1
        totals[1] += 1 if left == right else 0
    return {
        "name": scene["name"],
        "shape": list(gray.shape[:2]),
        "params": params,
        "noise": float(scene.get("noise", 0.0)),
        "reference_pass": bool(reference["pass"]),
        "device_pass": bool(device["pass"]),
        "reference_defects": len(reference["defects"]),
        "device_defects": len(device["defects"]),
        "identical": not differences,
        "differences": differences,
        "field_equality": field_equality,
        "mask_bit_identical": mask_equal,
        "residual_metadata_deltas": residual_deltas,
        "gaussian_calls": blur_calls,
        "gaussian_sigma": float(params.get("gaussian_sigma", 0.0)),
        "device_median_calls": median_calls,
        "reference_backend": reference["execution"]["backend"],
        "device_backend": device["execution"]["backend"],
    }


def compared_fields(reference: dict, device: dict):
    """Yield (key, reference_value, device_value) for every field the strict comparison covers.

    ``key`` is index-free so the caller can aggregate per-field equality over all cases; the
    differing-value report below re-adds the defect index for readability.
    """
    yield "pass", reference["pass"], device["pass"]
    yield "defect_count", len(reference["defects"]), len(device["defects"])
    for left, right in zip(reference["defects"], device["defects"]):
        for field in STRUCTURAL_DEFECT_FIELDS:
            yield f"defects.{field}", left.get(field), right.get(field)
        left_metadata = left.get("metadata", {})
        right_metadata = right.get("metadata", {})
        for key in sorted(set(left_metadata) | set(right_metadata)):
            yield f"defects.metadata.{key}", left_metadata.get(key), right_metadata.get(key)


def compare_results(reference: dict, device: dict) -> list[dict]:
    """Strict comparison; returns one entry per differing field."""
    differences: list[dict] = []
    for index, (left, right) in enumerate(zip(reference["defects"], device["defects"])):
        for field in STRUCTURAL_DEFECT_FIELDS:
            if left.get(field) != right.get(field):
                differences.append({
                    "field": f"defects[{index}].{field}",
                    "reference": left.get(field), "device": right.get(field),
                })
        left_metadata = left.get("metadata", {})
        right_metadata = right.get("metadata", {})
        for key in sorted(set(left_metadata) | set(right_metadata)):
            if left_metadata.get(key) != right_metadata.get(key):
                differences.append({
                    "field": f"defects[{index}].metadata.{key}",
                    "reference": left_metadata.get(key), "device": right_metadata.get(key),
                })
    if bool(reference["pass"]) != bool(device["pass"]):
        differences.append({"field": "pass", "reference": reference["pass"], "device": device["pass"]})
    if len(reference["defects"]) != len(device["defects"]):
        differences.append({
            "field": "defect_count",
            "reference": len(reference["defects"]),
            "device": len(device["defects"]),
        })
    return differences


def summarize_residual_deltas(reference: dict, device: dict) -> dict:
    """Largest absolute difference per residual-derived metadata field (diagnostic, not a gate)."""
    summary: dict[str, dict] = {}
    for index, (left, right) in enumerate(zip(reference["defects"], device["defects"])):
        for field in RESIDUAL_METADATA_FIELDS:
            left_value = left.get("metadata", {}).get(field)
            right_value = right.get("metadata", {}).get(field)
            if not isinstance(left_value, (int, float)) or not isinstance(right_value, (int, float)):
                continue
            delta = abs(float(left_value) - float(right_value))
            entry = summary.setdefault(field, {"max_abs": 0.0, "exact": True})
            entry["max_abs"] = max(entry["max_abs"], delta)
            entry["exact"] = entry["exact"] and delta == 0.0
            entry["defect_index"] = index
    return summary


def structural_only(result: dict) -> bool:
    """True when every difference (if any) is a residual-derived metadata float."""
    for difference in result["differences"]:
        field = difference["field"]
        if field in ("pass", "defect_count"):
            continue
        if ".metadata." in field and field.rsplit(".", 1)[1] in RESIDUAL_METADATA_FIELDS:
            continue
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", default=str(ROOT / "gpu" / "visionflow_cuda.dll"))
    parser.add_argument("--only", default="", help="comma separated scene names (default: all)")
    args = parser.parse_args()

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    runtime = GpuRuntime(args.dll)
    if not runtime.available:
        print(f"CUDA runtime unavailable: {runtime.unavailable_reason}")
        return 2
    if not runtime.supports_gaussian_blur_f32:
        print("CUDA DLL has no vf_gaussian_blur_f32 export; rebuild with gpu\\build_cuda_dll.ps1")
        return 2

    import detectors.detector_202_1 as detector_module

    scenes = build_scenes()
    if args.only:
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        scenes = [scene for scene in scenes if scene["name"] in wanted]

    results = []
    for scene in scenes:
        result = run_case(scene, runtime, detector_module)
        results.append(result)
        status = "identical" if result["identical"] else "DIFFERS"
        print(
            f"{result['name']:28s} {str(result['shape']):12s} pass={result['device_pass']!s:5s} "
            f"defects={result['device_defects']:3d} mask_equal={result['mask_bit_identical']!s:5s} "
            f"{status}"
        )
        for difference in result["differences"]:
            print(f"    {difference['field']}: reference={difference['reference']!r} device={difference['device']!r}")

    identical = sum(1 for result in results if result["identical"])
    mask_identical = sum(1 for result in results if result["mask_bit_identical"])
    pass_identical = sum(
        1 for result in results if result["reference_pass"] == result["device_pass"]
    )
    count_identical = sum(
        1 for result in results if result["reference_defects"] == result["device_defects"]
    )
    structural_identical = sum(1 for result in results if structural_only(result))
    device_calls = sum(
        1 for result in results for call in result["gaussian_calls"] if call[0] == "device"
    )
    cpu_calls = sum(
        1 for result in results for call in result["gaussian_calls"] if call[0] == "cpu"
    )
    backends = sorted({
        result["reference_backend"] for result in results
    } | {result["device_backend"] for result in results})

    metadata_deltas: dict[str, float] = {}
    for result in results:
        for field, entry in result["residual_metadata_deltas"].items():
            metadata_deltas[field] = max(metadata_deltas.get(field, 0.0), float(entry["max_abs"]))

    lines: list[str] = []
    lines.append("== 202-CS-SN-1 final-output matrix: cv2.GaussianBlur vs vf_gaussian_blur_f32 ==")
    lines.append(f"device: {runtime.device_name} (sm {runtime.compute_capability})")
    lines.append(f"scenes: {len(results)} unified/whole-pipeline runs per mode, both with use_gpu=True")
    lines.append(f"gaussian call sites substituted: device={device_calls}, left on cv2={cpu_calls}")
    sigma_calls = sorted({
        call[2] for result in results for call in result["gaussian_calls"] if call[0] == "device"
    })
    sigma_scenes = [result for result in results if result["gaussian_sigma"] != 0.0]
    lines.append(f"GaussianBlur sigma values forwarded to the device: {sigma_calls}")
    lines.append(
        f"scenes with a non-zero gaussian_sigma recipe parameter: {len(sigma_scenes)} "
        f"({[result['name'] for result in sigma_scenes]})"
    )
    lines.append(
        "gpu exact-median calls per run: "
        f"min reference={min(result['device_median_calls']['reference'] for result in results)}, "
        f"min device={min(result['device_median_calls']['device'] for result in results)}"
    )
    lines.append(f"detector backends observed: {backends}")
    lines.append(f"reference defects over all scenes: {sum(r['reference_defects'] for r in results)}")
    lines.append("")
    lines.append("[strict comparison] pass, defect count, and every defect's type/bbox_local/area/")
    lines.append("                    confidence/complete metadata mapping")
    lines.append(f"  final-output identical: {identical}/{len(results)}")
    lines.append(f"  PASS/NG identical: {pass_identical}/{len(results)}")
    lines.append(f"  defect count identical: {count_identical}/{len(results)}")
    lines.append(f"  structural fields identical (pass/count/type/bbox/area/confidence): "
                 f"{structural_identical}/{len(results)}")
    lines.append(f"  candidate mask bit-identical: {mask_identical}/{len(results)}")
    lines.append("")
    lines.append("[per-field equality] every compared field, over all cases")
    field_totals: dict[str, list[int]] = {}
    for result in results:
        for key, (compared, equal) in result["field_equality"].items():
            totals = field_totals.setdefault(key, [0, 0])
            totals[0] += compared
            totals[1] += equal
    for key in sorted(field_totals):
        compared, equal = field_totals[key]
        marker = "" if equal == compared else "   <-- differs"
        lines.append(f"  {key:44s} {equal}/{compared}{marker}")
    lines.append("")
    lines.append("[per-case]")
    for result in results:
        lines.append(
            f"  {result['name']:32s} shape={str(tuple(result['shape'])):12s} "
            f"sigma={result['gaussian_sigma']:<5} "
            f"pass={result['device_pass']!s:5s} defects={result['device_defects']:3d} "
            f"mask_bit_equal={result['mask_bit_identical']!s:5s} "
            f"strict_identical={result['identical']!s:5s}"
        )
    lines.append("")
    lines.append("[residual-derived metadata] largest absolute difference over every defect")
    if metadata_deltas:
        for field in sorted(metadata_deltas):
            lines.append(f"  {field:24s} max|delta| = {metadata_deltas[field]:.6e}")
    else:
        lines.append("  (no defects in any scene)")
    lines.append("")
    lines.append("[differences] every field that was not strictly equal")
    if not any(result["differences"] for result in results):
        lines.append("  none")
    for result in results:
        for difference in result["differences"]:
            lines.append(
                f"  {result['name']}: {difference['field']} "
                f"reference={difference['reference']!r} device={difference['device']!r}"
            )

    record = {
        "device": runtime.device_name,
        "compute_capability": runtime.compute_capability,
        "scenes": len(results),
        "identical": identical,
        "pass_identical": pass_identical,
        "count_identical": count_identical,
        "structural_identical": structural_identical,
        "mask_bit_identical": mask_identical,
        "device_gaussian_calls": device_calls,
        "cpu_gaussian_calls": cpu_calls,
        "backends": backends,
        "residual_metadata_max_delta": metadata_deltas,
        "field_equality": {
            key: {"compared": value[0], "equal": value[1]}
            for key, value in sorted(field_totals.items())
        },
        "cases": results,
    }

    EVIDENCE_TEXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    EVIDENCE_JSON.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print()
    print("\n".join(lines))
    print(f"\nwritten: {EVIDENCE_TEXT.relative_to(ROOT)}")
    print(f"written: {EVIDENCE_JSON.relative_to(ROOT)}")
    runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
