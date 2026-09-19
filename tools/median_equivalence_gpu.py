"""Exact GPU median: `vf_median_f32` equivalence against `np.median` on float32 input.

The CUDA export maps every float32 bit pattern to a monotone unsigned key, radix-sorts those keys
and reads back only the one or two middle keys; the float32 average for an even count is computed
on the host in plain C. No device-side floating-point arithmetic is involved, so this file holds a
NumPy golden reference (a mirror of exactly that pipeline) and requires the bridge result to be
`==` `np.median` for every contract case.

Usage (RTX 3090 evidence). The default `--dll` is an explicit build staging path so this script
never depends on the shared, frequently rebuilt `gpu/visionflow_cuda.dll`:

    .\\env\\Scripts\\python.exe tools\\median_equivalence_gpu.py ^
        --dll outputs_validation\\cuda_build_median\\visionflow_cuda.dll
    .\\env\\Scripts\\python.exe tools\\median_equivalence_gpu.py --golden-only

Evidence is written under outputs_validation/cnr_profile/.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# The reference itself overflows for the deliberately extreme even-count cases; those cases are
# contract checks, not defects, so keep the evidence output free of RuntimeWarning noise.
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.gpu_runtime import GpuRuntime  # noqa: E402

OUTPUT = ROOT / "outputs_validation" / "cnr_profile"
DEFAULT_DLL = ROOT / "outputs_validation" / "cuda_build_median" / "visionflow_cuda.dll"
SIGN_BIT = np.uint32(0x80000000)
COUNT_MASK = np.uint32(0xFFFFFFFF)
TIMING_SIZE = 4_000_000


def sortable_keys(values: np.ndarray) -> np.ndarray:
    """Monotone float32 -> uint32 key map; the device transform uses the same rule."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    negative = (bits & SIGN_BIT) != 0
    return np.where(negative, ~bits & COUNT_MASK, bits | SIGN_BIT).astype(np.uint32)


def key_to_value(key: int) -> np.float32:
    """Inverse of sortable_keys() for one key, as a float32 scalar."""
    key = int(key) & 0xFFFFFFFF
    bits = (key & 0x7FFFFFFF) if (key & 0x80000000) else ((~key) & 0xFFFFFFFF)
    return np.array([bits], dtype=np.uint32).view(np.float32)[0]


def golden_median(values: np.ndarray) -> np.float32:
    """NumPy mirror of the native pipeline, including numpy's even-count float32 average."""
    keys = np.sort(sortable_keys(values))
    count = int(keys.size)
    middle = count // 2
    if count % 2 == 1:
        return key_to_value(int(keys[middle]))
    low = key_to_value(int(keys[middle - 1]))
    high = key_to_value(int(keys[middle]))
    # np.median uses mean() of the two middle float32 values: a float32 add, then a float32
    # divide by two (an exact power-of-two scaling). Mirror it exactly.
    with np.errstate(all="ignore"):
        return np.float32(np.float32(low + high) / np.float32(2.0))


def _cnr_like(rng: np.random.Generator, count: int) -> np.ndarray:
    """Zero-centred normal residual (sigma about 6) with a few large positive outliers."""
    values = rng.normal(0.0, 6.0, count).astype(np.float32)
    outlier_count = max(1, count // 200_000)
    positions = rng.integers(0, count, outlier_count)
    values[positions] = rng.uniform(40.0, 200.0, outlier_count).astype(np.float32)
    return values


def _extreme_pool() -> np.ndarray:
    info = np.finfo(np.float32)
    return np.array(
        [
            info.max,
            -info.max,
            np.float32(1.0e38),
            np.float32(-1.0e38),
            info.tiny,
            -info.tiny,
            info.smallest_subnormal,
            -info.smallest_subnormal,
            np.float32(0.0),
            np.float32(-0.0),
            np.float32(1.0),
            np.float32(-1.0),
        ],
        dtype=np.float32,
    )


def build_cases() -> list[tuple[str, np.ndarray]]:
    """The required equivalence matrix plus extra boundary cases that must not regress."""
    rng = np.random.default_rng(20260914)
    pool = _extreme_pool()
    cases: list[tuple[str, np.ndarray]] = []
    # Required: sizes 1, 2, 3, 11, 1000, 400001, 2000001, 8000001.
    for count in (1, 2, 3, 11, 1000, 400_001, 2_000_001, 8_000_001):
        cases.append((f"normal_{count}", rng.normal(0.0, 6.0, count).astype(np.float32)))
    # Required: all-equal, all-zero (both parities so the even-count average is exercised).
    cases.append(("all_equal_400001", np.full(400_001, np.float32(7.5), dtype=np.float32)))
    cases.append(("all_equal_400000", np.full(400_000, np.float32(-3.25), dtype=np.float32)))
    cases.append(("all_zero_400001", np.zeros(400_001, dtype=np.float32)))
    cases.append(("all_zero_400000", np.zeros(400_000, dtype=np.float32)))
    # Required: all-negative.
    cases.append(("all_negative_400001", (-np.abs(rng.normal(0.0, 5.0, 400_001))).astype(np.float32)))
    cases.append(("all_negative_400000", (-np.abs(rng.normal(0.0, 5.0, 400_000))).astype(np.float32)))
    # Required: CNR-like residual.
    cases.append(("cnr_like_400001", _cnr_like(rng, 400_001)))
    cases.append(("cnr_like_400000", _cnr_like(rng, 400_000)))
    cases.append(("cnr_like_8000001", _cnr_like(rng, 8_000_001)))
    # Required: values spanning a wide but finite float32 range (no infinities, no NaNs).
    cases.append(("extreme_range_400001", pool[rng.integers(0, pool.size, 400_001)].copy()))
    cases.append(("extreme_range_400000", pool[rng.integers(0, pool.size, 400_000)].copy()))
    cases.append(("extreme_range_2", np.array([pool[0], pool[1]], dtype=np.float32)))
    # Explicitly force the even-count average onto the float32 boundaries: an overflowing sum,
    # an exactly cancelling sum, and subnormal results rounded by true division.
    info = np.finfo(np.float32)
    cases.append(("extreme_max_pair_2", np.array([info.max, info.max], dtype=np.float32)))
    cases.append(("extreme_max_cancel_2", np.array([info.max, -info.max], dtype=np.float32)))
    cases.append(("subnormal_pair_2", np.array(
        [info.smallest_subnormal, info.smallest_subnormal], dtype=np.float32)))
    cases.append(("subnormal_half_2", np.array(
        [np.float32(0.0), info.smallest_subnormal], dtype=np.float32)))
    cases.append(("subnormal_half_3", np.array(
        [np.float32(0.0), info.smallest_subnormal, info.smallest_subnormal], dtype=np.float32)))
    cases.append(("tiny_pair_2", np.array([info.tiny, info.tiny], dtype=np.float32)))
    cases.append(("signed_zero_400000", pool[rng.integers(8, 10, 400_000)].copy()))
    cases.append(("signed_zero_400001", pool[rng.integers(8, 10, 400_001)].copy()))
    cases.append(("ascending_400000", np.sort(rng.normal(0.0, 3.0, 400_000).astype(np.float32))))
    cases.append(("descending_400001", np.sort(rng.normal(0.0, 3.0, 400_001).astype(np.float32))[::-1].copy()))
    cases.append(("two_values_1000000", np.where(
        rng.integers(0, 2, 1_000_000) == 0, np.float32(-1.5), np.float32(2.25)).astype(np.float32)))
    duplicates = rng.normal(0.0, 1.0, 1024).astype(np.float32)
    cases.append(("duplicates_1000000", duplicates[rng.integers(0, duplicates.size, 1_000_000)].copy()))
    # Infinities follow the same total order, so they are part of the exact contract.
    cases.append(("infinity_4", np.array([1.0, np.inf, -np.inf, 3.0], dtype=np.float32)))
    cases.append(("infinity_3", np.array([1.0, np.inf, 3.0], dtype=np.float32)))
    return cases


def build_nan_cases() -> list[tuple[str, np.ndarray]]:
    """Informational only: ``NaN != NaN``, so these are compared with a NaN-aware test."""
    rng = np.random.default_rng(31337)
    return [
        ("nan_3", np.array([1.0, 2.0, np.nan], dtype=np.float32)),
        ("nan_4", np.array([1.0, 2.0, 3.0, np.nan], dtype=np.float32)),
        ("nan_negative", np.array([-np.nan, 1.0, 2.0], dtype=np.float32)),
        ("nan_few_in_1003", np.concatenate(
            [rng.normal(0.0, 6.0, 1000).astype(np.float32), np.full(3, np.nan, np.float32)])),
        ("nan_all_11", np.full(11, np.nan, np.float32)),
        ("nan_with_infinity_4", np.array([1.0, np.inf, np.nan, 3.0], dtype=np.float32)),
    ]


def same_result(produced: np.float32, reference: np.float32) -> tuple[bool, bool]:
    """Return (equal, both_nan_or_equal). ``equal`` is the plain ``==`` contract."""
    equal = bool(produced == reference)
    both_nan = bool(np.isnan(produced) and np.isnan(reference))
    return equal, bool(equal or both_nan)


def compare(name: str, produced: np.float32, reference: np.float32) -> dict:
    produced = np.float32(produced)
    equal, matches = same_result(produced, reference)
    same_bits = bool(produced.tobytes() == reference.tobytes())
    with np.errstate(all="ignore"):
        difference = float(np.abs(np.float64(produced) - np.float64(reference)))
    return {
        "case": name,
        "produced": float(produced),
        "reference": float(reference),
        "identical": equal,
        "nan_match": matches,
        "bits_identical": same_bits,
        "abs_error": difference if np.isfinite(difference) else None,
    }


def run_golden(cases: list[tuple[str, np.ndarray]], report: dict) -> int:
    print("NumPy golden mirror versus np.median")
    rows = []
    failures = 0
    for name, values in cases:
        row = compare(name, golden_median(values), np.median(values))
        rows.append(row)
        if not row["nan_match"]:
            failures += 1
        print(
            f"  {name:<22} mirror={row['produced']!r:<24} numpy={row['reference']!r:<24} "
            f"identical={row['identical']}"
        )
    report["golden_mirror"] = rows
    report["golden_mirror_failures"] = failures
    print(f"golden mirror identical: {len(rows) - failures}/{len(rows)}")
    return failures


def run_gpu(
    runtime: GpuRuntime,
    cases: list[tuple[str, np.ndarray]],
    report: dict,
    key: str = "gpu_median",
    strict_equality: bool = True,
) -> int:
    print("vf_median_f32 versus np.median")
    rows = []
    failures = 0
    for name, values in cases:
        produced = runtime.median_f32(values)
        row = compare(name, produced, np.median(values))
        row["count"] = int(values.size)
        row["result_dtype"] = str(np.asarray(produced).dtype)
        ok = row["identical"] if strict_equality else row["nan_match"]
        if not ok or row["result_dtype"] != "float32":
            failures += 1
        rows.append(row)
        print(
            f"  {name:<22} count={row['count']:<9} gpu={row['produced']!r:<24} "
            f"numpy={row['reference']!r:<24} identical={row['identical']} "
            f"bits={row['bits_identical']} abs_err={row['abs_error']!r}"
        )
    report[key] = rows
    report[f"{key}_failures"] = failures
    print(f"{key} identical={len(rows) - failures}/{len(rows)}")
    return failures


def run_stability(runtime: GpuRuntime, cases: list[tuple[str, np.ndarray]], report: dict) -> int:
    """Prove the operand is never mutated and that repeated calls are byte-deterministic."""
    print("operand non-mutation and determinism")
    rows = []
    failures = 0
    for name, values in cases:
        before = values.tobytes()
        first = np.float32(runtime.median_f32(values))
        after = values.tobytes()
        second = np.float32(runtime.median_f32(values))
        mutated = after != before
        deterministic = first.tobytes() == second.tobytes()
        if mutated or not deterministic:
            failures += 1
        rows.append(
            {
                "case": name,
                "count": int(values.size),
                "operand_mutated": bool(mutated),
                "deterministic": bool(deterministic),
            }
        )
    report["stability"] = rows
    report["stability_failures"] = failures
    print(
        f"  cases={len(rows)} mutated={sum(1 for row in rows if row['operand_mutated'])} "
        f"non_deterministic={sum(1 for row in rows if not row['deterministic'])}"
    )
    return failures


def run_timing(runtime: GpuRuntime, size: int, repeats: int, report: dict) -> None:
    rng = np.random.default_rng(4242)
    values = rng.normal(0.0, 6.0, size).astype(np.float32)
    reference = np.median(values)
    runtime.median_f32(values)  # warm up the context, cub scratch, and the sort policy
    gpu_samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = runtime.median_f32(values)
        gpu_samples.append((time.perf_counter() - started) * 1000.0)
        if not bool(np.float32(result) == reference):
            raise AssertionError(f"timed call diverged from np.median at size {size}")
    cpu_samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        np.median(values)
        cpu_samples.append((time.perf_counter() - started) * 1000.0)
    gpu_median_ms = statistics.median(gpu_samples)
    cpu_median_ms = statistics.median(cpu_samples)
    native = runtime.performance_stats().get("native_timings_ms") or {}
    report["timing"] = {
        "count": int(size),
        "repeats": int(repeats),
        "gpu_median_ms": round(gpu_median_ms, 3),
        "gpu_min_ms": round(min(gpu_samples), 3),
        "gpu_p95_ms": round(float(np.percentile(gpu_samples, 95)), 3),
        "cpu_median_ms": round(cpu_median_ms, 3),
        "cpu_min_ms": round(min(cpu_samples), 3),
        "speedup": round(cpu_median_ms / gpu_median_ms, 3) if gpu_median_ms > 0 else None,
        "native_timings_ms": native,
    }
    print(
        f"{size:,} float32 values: GPU {gpu_median_ms:.2f} ms | CPU np.median {cpu_median_ms:.2f} ms "
        f"| speedup {cpu_median_ms / gpu_median_ms:.2f}x"
    )
    print(
        "  native CUDA events: "
        f"h2d={native.get('h2d_ms')} kernel={native.get('kernel_ms')} "
        f"d2h={native.get('d2h_ms')} total={native.get('total_device_ms')}"
    )


def render_text(report: dict) -> str:
    lines = [
        f"device: {report.get('device', '')} ({report.get('compute_capability', '')})",
        f"dll: {report.get('dll', '')}",
        f"golden mirror identical: {report.get('golden_identical', 0)}/{report.get('golden_total', 0)}",
        f"vf_median_f32 identical: {report.get('gpu_identical', 0)}/{report.get('gpu_total', 0)}",
        f"nan cases nan-match: {report.get('nan_identical', 0)}/{report.get('nan_total', 0)}",
        f"stability failures: {report.get('stability_failures', 0)}/{report.get('stability_total', 0)}",
    ]
    timing = report.get("timing")
    if timing:
        lines.append(
            f"{timing['count']:,} float32 values: GPU {timing['gpu_median_ms']:.2f} ms "
            f"(P95 {timing['gpu_p95_ms']:.2f}, min {timing['gpu_min_ms']:.2f}) | "
            f"CPU np.median {timing['cpu_median_ms']:.2f} ms | speedup {timing['speedup']:.2f}x"
        )
    for row in report.get("gpu_median", []):
        lines.append(
            f"{row['case']:<22} count={row['count']:<9} gpu={row['produced']!r:<24} "
            f"numpy={row['reference']!r:<24} identical={row['identical']} "
            f"bits_identical={row['bits_identical']}"
        )
    for row in report.get("nan_cases", []):
        lines.append(
            f"{row['case']:<22} count={row['count']:<9} gpu={row['produced']!r:<24} "
            f"numpy={row['reference']!r:<24} identical={row['identical']} nan_match={row['nan_match']}"
        )
    lines.append(f"identical={report.get('gpu_identical', 0)}/{report.get('gpu_total', 0)}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare vf_median_f32 with np.median on float32.")
    parser.add_argument("--dll", type=Path, default=DEFAULT_DLL)
    parser.add_argument("--golden-only", action="store_true", help="Only run the NumPy mirror.")
    parser.add_argument("--timing-size", type=int, default=TIMING_SIZE)
    parser.add_argument("--timing-repeats", type=int, default=11)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    nan_cases = build_nan_cases()
    report: dict = {"cases": len(cases), "nan_case_count": len(nan_cases), "timing_size": int(args.timing_size)}
    failures = run_golden(cases, report)
    report["golden_total"] = len(cases)
    report["golden_identical"] = len(cases) - failures

    if not args.golden_only:
        runtime = GpuRuntime(str(args.dll), fallback_to_cpu=True)
        report["dll"] = str(args.dll)
        report["device"] = runtime.device_name
        report["compute_capability"] = runtime.compute_capability
        if not runtime.available or not runtime.supports_exact_median:
            print("vf_median_f32 is unavailable:", runtime.unavailable_reason or "missing export")
            report["unavailable"] = runtime.unavailable_reason or "missing export"
            failures += 1
        else:
            median_failures = run_gpu(runtime, cases, report)
            report["gpu_total"] = len(cases)
            report["gpu_identical"] = len(cases) - median_failures
            failures += median_failures
            nan_failures = run_gpu(
                runtime, nan_cases, report, key="nan_cases", strict_equality=False)
            report["nan_total"] = len(nan_cases)
            report["nan_identical"] = len(nan_cases) - nan_failures
            failures += nan_failures
            stability_failures = run_stability(runtime, cases + nan_cases, report)
            report["stability_total"] = len(cases) + len(nan_cases)
            failures += stability_failures
            run_timing(runtime, int(args.timing_size), int(args.timing_repeats), report)
        runtime.close()

    json_path = args.output / "median_gpu_equivalence.json"
    text_path = args.output / "median_gpu_equivalence.txt"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    text_path.write_text(render_text(report), encoding="utf-8")
    print(text_path)
    print(json_path)
    if failures:
        print(f"FAILURES: {failures}")
        return 1
    print("ALL IDENTICAL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
