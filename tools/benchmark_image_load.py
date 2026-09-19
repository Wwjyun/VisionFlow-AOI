"""Compare ImageLoader against cv2.imdecode on one image: pixel identity and median load time.

The production images are 24-bit BMP. ``core.image_loader.BmpReader`` reads them in parallel row bands;
this tool re-measures that against the OpenCV reference on the machine it runs on and fails when the
pixels differ.

Usage:
    .\\env\\Scripts\\python.exe tools\\benchmark_image_load.py --image <path.bmp> [--repetitions 5]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.image_loader import BmpReader, ImageLoader  # noqa: E402


def _median_ms(function, repetitions: int):
    samples = []
    result = None
    for _ in range(repetitions + 1):
        started = time.perf_counter()
        result = function()
        samples.append((time.perf_counter() - started) * 1000.0)
    measured = samples[1:]
    return result, statistics.median(measured), max(measured)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    path = args.image
    reference, reference_ms, reference_max = _median_ms(
        lambda: cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR), args.repetitions
    )
    loader = ImageLoader()
    loaded, loader_ms, loader_max = _median_ms(lambda: loader.load_bgr(path), args.repetitions)
    fast_path = BmpReader().read(path) is not None if path.suffix.lower() == ".bmp" else False
    identical = bool(reference is not None and np.array_equal(loaded, reference))
    payload = {
        "image": str(path),
        "shape": list(loaded.shape),
        "file_mb": path.stat().st_size / 1e6,
        "bmp_reader_used": fast_path,
        "identical_to_opencv": identical,
        "opencv_median_ms": reference_ms,
        "opencv_max_ms": reference_max,
        "image_loader_median_ms": loader_ms,
        "image_loader_max_ms": loader_max,
        "speedup": reference_ms / loader_ms if loader_ms else None,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for key, value in payload.items():
        print(f"{key}: {value}")
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
