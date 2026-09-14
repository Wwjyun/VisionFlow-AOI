"""Acceptance gate for the GPU contour operator: compare it with OpenCV point-for-point.

This is the check that must pass before the CUDA contour path may replace the CPU
``cv2.findContours`` in a detector, so it is deliberately stricter than a similarity test: it
compares the contour count, every contour's shape, and every point in order, for both contour
modes. While the operator is not implemented yet the script reports that clearly and exits 0, so it
can live in the tree as the ready-made gate.

Usage: .\\env\\Scripts\\python.exe tools/check_contour_equivalence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.gpu_runtime import GpuRuntime, GpuRuntimeError  # noqa: E402

DLL = ROOT / "gpu" / "visionflow_cuda.dll"
OUTPUT = ROOT / "outputs_validation" / "contour_equivalence"

MODES = (("list", cv2.RETR_LIST), ("external", cv2.RETR_EXTERNAL))


def cases() -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(20260914)
    shapes = []
    solid = np.zeros((64, 80), dtype=np.uint8)
    solid[12:34, 18:52] = 255
    shapes.append(("solid_rect", solid))

    two = solid.copy()
    two[44:56, 60:76] = 255
    shapes.append(("two_rects", two))

    ring = np.zeros((64, 80), dtype=np.uint8)
    ring[10:50, 14:64] = 255
    ring[20:40, 26:52] = 0
    shapes.append(("ring_hole", ring))

    stairs = np.zeros((64, 80), dtype=np.uint8)
    for row in range(10, 50):
        stairs[row, 8 : 8 + (row - 9)] = 255
    shapes.append(("staircase", stairs))

    lines = np.zeros((64, 80), dtype=np.uint8)
    lines[30, 5:70] = 255
    lines[5:40, 40] = 255
    lines[55, 55] = 255
    shapes.append(("thin_lines", lines))

    border = np.zeros((64, 80), dtype=np.uint8)
    border[0:6, 0:6] = 255
    border[:, -1] = 255
    border[-1, :] = 255
    shapes.append(("touching_border", border))

    shapes.append(("all_zero", np.zeros((64, 80), dtype=np.uint8)))
    shapes.append(("all_full", np.full((64, 80), 255, dtype=np.uint8)))

    nested = np.zeros((64, 80), dtype=np.uint8)
    nested[6:58, 8:72] = 255
    nested[14:50, 18:62] = 0
    nested[22:42, 28:52] = 255
    shapes.append(("nested_rings", nested))

    for seed in (1, 2, 3, 4, 5):
        random_mask = (rng.integers(0, 100, (64, 80)) > 80).astype(np.uint8) * 255
        shapes.append((
            f"random_open_{seed}",
            cv2.morphologyEx(random_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)),
        ))
    return shapes


def compare(reference: list[np.ndarray], actual: list[np.ndarray]) -> tuple[bool, str]:
    if len(reference) != len(actual):
        return False, f"contour count cv2={len(reference)} gpu={len(actual)}"
    for index, (expected, mine) in enumerate(zip(reference, actual)):
        if expected.shape != mine.shape:
            return False, (f"contour {index} shape cv2={expected.shape} gpu={mine.shape}")
        if not np.array_equal(expected, mine):
            difference = int(np.argwhere(expected.reshape(-1, 2) != mine.reshape(-1, 2))[0][0])
            return False, (
                f"contour {index} first differing point {difference}: "
                f"cv2={expected.reshape(-1, 2)[difference].tolist()} "
                f"gpu={mine.reshape(-1, 2)[difference].tolist()}"
            )
    return True, "identical"


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runtime = GpuRuntime(str(DLL), fallback_to_cpu=True)
    export = getattr(runtime._dll, "vf_find_contours_u8", None) if runtime._dll is not None else None
    if not runtime.available or export is None:
        message = (
            "GPU contour operator not available yet "
            f"(dll_loaded={runtime._dll is not None}, export_present={export is not None}). "
            "This gate becomes active as soon as vf_find_contours_u8 exists."
        )
        print(message)
        (OUTPUT / "contour_equivalence_status.txt").write_text(message + "\n", encoding="utf-8")
        runtime.close()
        return 0

    identical = 0
    total = 0
    rows = []
    for name, mask in cases():
        for mode, flag in MODES:
            reference, _ = cv2.findContours(mask, flag, cv2.CHAIN_APPROX_SIMPLE)
            try:
                actual = runtime.find_contours_gray(mask, mode)
            except GpuRuntimeError as exc:
                rows.append(f"{name}/{mode}: GPU error {exc}")
                total += 1
                continue
            same, detail = compare(reference, actual)
            identical += int(same)
            total += 1
            rows.append(f"{name}/{mode}: {'identical' if same else 'DIFFERS - ' + detail}")
            print(rows[-1])
    summary = f"identical {identical}/{total}"
    print(summary)
    (OUTPUT / "contour_equivalence.txt").write_text("\n".join([*rows, summary]) + "\n", encoding="utf-8")
    runtime.close()
    return 0 if identical == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
