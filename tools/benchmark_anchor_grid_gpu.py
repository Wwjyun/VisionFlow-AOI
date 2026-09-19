"""RTX 3090 benchmark: GPU vs CPU Template Anchor Grid localization.

Reads the same search-ROI shape the production recipes use and reports both backends with warm
medians, plus the equivalence of every result. Writes JSON evidence under outputs_validation/.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.gpu_runtime import GpuRuntime  # noqa: E402
from core.tiler import GridAnchorConfig, Tiler  # noqa: E402

OUTPUT = ROOT / "outputs_validation" / "anchor_grid_gpu"
DLL = ROOT / "gpu" / "visionflow_cuda.dll"


def build_scene(height: int, width: int, template_size: tuple[int, int], seed: int = 7):
    """A structured board with one unique anchor patch, plus a gray template of that patch."""
    rng = np.random.default_rng(seed)
    board = (rng.integers(0, 40, (height, width, 3))).astype(np.uint8)
    for row in range(0, height, 48):
        for column in range(0, width, 48):
            board[row : row + 24, column : column + 24] = (
                (row * 5 + column * 7) % 180 + 40
            )
    template_width, template_height = template_size
    anchor_x = int(width * 0.35)
    anchor_y = int(height * 0.30)
    patch = np.zeros((template_height, template_width, 3), dtype=np.uint8)
    for row in range(template_height):
        for column in range(template_width):
            patch[row, column] = ((row * 13 + column * 29 + seed * 5) % 190 + 30)
    board[anchor_y : anchor_y + template_height, anchor_x : anchor_x + template_width] = patch
    return board, patch, (anchor_x, anchor_y)


def time_call(function, repetitions: int) -> dict:
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        function()
        samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runtime = GpuRuntime(str(DLL), fallback_to_cpu=True)
    if not runtime.available or not runtime.supports_template_match:
        print("CUDA Template Anchor Grid localization is unavailable:", runtime.unavailable_reason)
        return 1
    print("device:", runtime.device_name, runtime.compute_capability)

    scenarios = [
        ("search_2000x12000_t300", (12000, 2000), (300, 300)),
        ("search_2000x12000_t80", (12000, 2000), (80, 80)),
        ("search_512x512_t80", (512, 512), (80, 80)),
    ]
    report = {"device": runtime.device_name, "compute_capability": runtime.compute_capability,
              "scenarios": []}
    failures = 0
    for name, (height, width), template_size in scenarios:
        board, patch, (anchor_x, anchor_y) = build_scene(height, width, template_size)
        template_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

        # CPU reference through the public tiler with the device disabled.
        cpu_tiler = Tiler(
            template_size[0], template_size[1],
            anchor_config=GridAnchorConfig.from_dict({
                "template_path": "template.png", "rows": 1, "cols": 1,
                "roi_w": template_size[0], "roi_h": template_size[1],
            }),
            gpu_runtime=None, resident_image=None,
        )
        template_path = OUTPUT / f"{name}_template.png"
        cv2.imwrite(str(template_path), patch)
        cpu_tiler.anchor_config = GridAnchorConfig.from_dict({
            "template_path": str(template_path), "rows": 1, "cols": 1,
            "roi_w": template_size[0], "roi_h": template_size[1],
        })
        cpu_anchor_result = cpu_tiler._find_grid_anchor(board, cpu_tiler.anchor_config)
        cpu_timing = time_call(
            lambda: cpu_tiler._find_grid_anchor(board, cpu_tiler.anchor_config), 3
        )

        resident = runtime.upload_image(board)
        gpu_tiler = Tiler(
            template_size[0], template_size[1],
            anchor_config=cpu_tiler.anchor_config,
            gpu_runtime=runtime, resident_image=resident,
            gpu_anchor_enabled=True,
        )
        if getattr(runtime, "supports_template_match", False):
            gpu_anchor_result = gpu_tiler._find_grid_anchor(board, gpu_tiler.anchor_config)
        else:
            gpu_anchor_result = {"x": -1, "y": -1, "width": 0, "height": 0, "score": 0.0,
                                 "backend": "unavailable"}
        gpu_timing = time_call(
            lambda: gpu_tiler._find_grid_anchor(board, gpu_tiler.anchor_config), 10
        )

        same_location = (cpu_anchor_result["x"], cpu_anchor_result["y"]) == (
            gpu_anchor_result["x"], gpu_anchor_result["y"]
        )
        expected_location = (anchor_x, anchor_y)
        if not same_location or (cpu_anchor_result["x"], cpu_anchor_result["y"]) != expected_location:
            failures += 1
        speedup = cpu_timing["median_ms"] / gpu_timing["median_ms"] if gpu_timing["median_ms"] else 0.0
        scenario = {
            "name": name,
            "search_shape": [height, width],
            "template_shape": list(template_size),
            "cpu": {**cpu_anchor_result, **cpu_timing, "backend": cpu_anchor_result.get("backend")},
            "gpu": {**gpu_anchor_result, **gpu_timing},
            "location_equal": bool(same_location),
            "location_matches_scene": (cpu_anchor_result["x"], cpu_anchor_result["y"]) == expected_location,
            "score_delta": abs(cpu_anchor_result["score"] - gpu_anchor_result["score"]),
            "speedup_cpu_over_gpu": speedup,
        }
        report["scenarios"].append(scenario)
        print(
            f"{name}: cpu median {cpu_timing['median_ms']:.1f} ms "
            f"({cpu_anchor_result['x']},{cpu_anchor_result['y']}) | "
            f"gpu median {gpu_timing['median_ms']:.1f} ms "
            f"({gpu_anchor_result['x']},{gpu_anchor_result['y']}) | "
            f"speedup {speedup:.2f}x | equal={same_location} | "
            f"score_delta={scenario['score_delta']:.3e}"
        )
        del board
    report["failures"] = failures
    evidence = OUTPUT / "anchor_grid_gpu_benchmark.json"
    evidence.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("evidence:", evidence)
    runtime.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
