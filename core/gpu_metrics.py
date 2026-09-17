from __future__ import annotations


_CUMULATIVE_FIELDS = (
    "load_sec",
    "call_count",
    "estimated_round_trips",
    "host_to_device_bytes",
    "device_to_host_bytes",
    "wall_sec",
    "lock_wait_sec",
    "kernel_launch_count",
)

_FUNCTION_CUMULATIVE_FIELDS = (
    "calls",
    "host_to_device_bytes",
    "device_to_host_bytes",
    "wall_sec",
    "lock_wait_sec",
)


def _nonnegative_difference(current, baseline, *, integer: bool = False):
    value = max(0, current - baseline)
    return int(value) if integer else round(float(value), 6)


def performance_stats_delta(current: dict, baseline: dict | None) -> dict:
    """Return metrics produced after ``baseline`` while preserving current runtime gauges.

    Runtime statistics are session-cumulative. Pipeline results need a per-run view so a call
    made by a warm-up or earlier image cannot make a later CPU route look like device work.
    """
    if not isinstance(current, dict):
        return {}
    before = baseline if isinstance(baseline, dict) else {}
    delta = {
        key: value
        for key, value in current.items()
        if key not in {*_CUMULATIVE_FIELDS, "native_cumulative_ms", "functions"}
    }
    for key in _CUMULATIVE_FIELDS:
        integer = key not in {"load_sec", "wall_sec", "lock_wait_sec"}
        delta[key] = _nonnegative_difference(
            current.get(key, 0) or 0,
            before.get(key, 0) or 0,
            integer=integer,
        )

    current_native = current.get("native_cumulative_ms") or {}
    before_native = before.get("native_cumulative_ms") or {}
    delta["native_cumulative_ms"] = {}
    for name, value in current_native.items():
        difference = _nonnegative_difference(value or 0, before_native.get(name, 0) or 0)
        if difference > 0:
            delta["native_cumulative_ms"][str(name)] = difference

    current_functions = current.get("functions") or {}
    before_functions = before.get("functions") or {}
    functions = {}
    for name, entry in current_functions.items():
        if not isinstance(entry, dict):
            continue
        previous = before_functions.get(name) or {}
        item = {}
        for key in _FUNCTION_CUMULATIVE_FIELDS:
            integer = key not in {"wall_sec", "lock_wait_sec"}
            item[key] = _nonnegative_difference(
                entry.get(key, 0) or 0,
                previous.get(key, 0) or 0,
                integer=integer,
            )
        if any(item.values()):
            functions[str(name)] = item
    delta["functions"] = functions
    if delta["call_count"] == 0:
        # The native DLL exposes only its most recent operation. Do not attach a prior image's
        # timing payload to a run that issued no CUDA call.
        delta["native_timings_ms"] = None
    delta["measurement_period"] = "current_pipeline_run"
    return delta


class GpuPerformanceRecorder:
    def __init__(self) -> None:
        self.values = {
            "load_sec": 0.0, "call_count": 0, "estimated_round_trips": 0,
            "host_to_device_bytes": 0, "device_to_host_bytes": 0,
            "wall_sec": 0.0, "lock_wait_sec": 0.0, "functions": {},
            "native_cumulative_ms": {}, "kernel_launch_count": 0,
            "peak_vram_bytes": 0,
        }

    def record(self, function_name: str, h2d: int, d2h: int, wall: float, wait: float) -> None:
        function = self.values["functions"].setdefault(function_name, {
            "calls": 0, "host_to_device_bytes": 0, "device_to_host_bytes": 0,
            "wall_sec": 0.0, "lock_wait_sec": 0.0,
        })
        function["calls"] += 1
        function["host_to_device_bytes"] += h2d
        function["device_to_host_bytes"] += d2h
        function["wall_sec"] += wall
        function["lock_wait_sec"] += wait
        self.values["call_count"] += 1
        self.values["estimated_round_trips"] += 1
        self.values["host_to_device_bytes"] += h2d
        self.values["device_to_host_bytes"] += d2h
        self.values["wall_sec"] += wall
        self.values["lock_wait_sec"] += wait

    def record_native(
        self,
        timings: dict | None,
        *,
        kernel_launch_count: int = 0,
        reserved_bytes: int = 0,
    ) -> None:
        if timings:
            cumulative = self.values["native_cumulative_ms"]
            for name, value in timings.items():
                if not isinstance(value, (int, float)):
                    continue
                if name == "context_create_ms":
                    cumulative[name] = max(float(cumulative.get(name, 0.0)), float(value))
                else:
                    cumulative[name] = float(cumulative.get(name, 0.0)) + float(value)
        self.values["kernel_launch_count"] += max(0, int(kernel_launch_count))
        self.values["peak_vram_bytes"] = max(
            int(self.values["peak_vram_bytes"]), max(0, int(reserved_bytes))
        )

    def snapshot(self) -> dict:
        values = self.values
        return {
            "load_sec": round(float(values["load_sec"]), 6),
            "call_count": int(values["call_count"]),
            "estimated_round_trips": int(values["estimated_round_trips"]),
            "host_to_device_bytes": int(values["host_to_device_bytes"]),
            "device_to_host_bytes": int(values["device_to_host_bytes"]),
            "wall_sec": round(float(values["wall_sec"]), 6),
            "lock_wait_sec": round(float(values["lock_wait_sec"]), 6),
            "native_cumulative_ms": {
                name: round(float(value), 6)
                for name, value in sorted(values["native_cumulative_ms"].items())
            },
            "kernel_launch_count": int(values["kernel_launch_count"]),
            "peak_vram_bytes": int(values["peak_vram_bytes"]),
            "functions": {
                name: {
                    "calls": int(item["calls"]),
                    "host_to_device_bytes": int(item["host_to_device_bytes"]),
                    "device_to_host_bytes": int(item["device_to_host_bytes"]),
                    "wall_sec": round(float(item["wall_sec"]), 6),
                    "lock_wait_sec": round(float(item["lock_wait_sec"]), 6),
                }
                for name, item in sorted(values["functions"].items())
            },
        }
