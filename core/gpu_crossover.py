from __future__ import annotations

import statistics
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass(slots=True)
class _CrossoverSamples:
    cuda_warmup_done: bool = False
    cuda_ms: list[float] = field(default_factory=list)
    cpu_ms: list[float] = field(default_factory=list)
    decision: str = ""


class PlanCrossoverPolicy:
    """Choose CPU or CUDA per preprocessing plan from measured cost on this machine.

    Small tiles and cheap operators can be slower on CUDA because every call pays a fixed
    D2D/launch/D2H/synchronize cost (RTX 3090, 2026-09-14: about 0.2 ms), while the CPU cost
    depends on the host. Each (plan signature, input shape, device ROI) key runs one CUDA
    warm-up, ``cuda_samples`` timed CUDA calls and ``cpu_samples`` timed CPU shadow calls, then
    freezes its decision for the runtime lifetime. CPU and CUDA plans are pixel-identical, so the
    route never changes inspection results. Only used when CPU fallback is allowed.
    """

    def __init__(
        self,
        cuda_samples: int = 3,
        cpu_samples: int = 2,
        margin: float = 1.15,
        max_entries: int = 256,
        clock=time.perf_counter,
    ):
        self.cuda_samples = max(1, int(cuda_samples))
        self.cpu_samples = max(1, int(cpu_samples))
        self.margin = max(1.0, float(margin))
        self.max_entries = max(1, int(max_entries))
        self.clock = clock
        self._entries: OrderedDict[tuple, _CrossoverSamples] = OrderedDict()
        self._resident_upload_unneeded: OrderedDict[tuple, None] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(plan, image, device_roi: bool) -> tuple:
        return (plan.signature, tuple(int(value) for value in image.shape), image.dtype.str, bool(device_roi))

    def decision(self, key: tuple) -> str:
        """Return ``"cpu"``, ``"cuda"`` or ``""`` while the key is still calibrating."""
        with self._lock:
            entry = self._entries.get(key)
            return entry.decision if entry is not None else ""

    @staticmethod
    def _resident_key(key: tuple) -> tuple:
        return (key[0], key[1], key[2], True)

    def prefer_cpu_key(self, key: tuple) -> tuple[bool, tuple]:
        """Decide one plan call from its recorded key, returning the decision key to report.

        A host-input CUDA call pays extra H2D on top of the resident ROI cost, so a CPU decision
        measured against the resident path also holds when the resident image was skipped. The
        reverse is not true: without an upload, only the resident measurement proves anything.
        """

        def field_of(candidate: tuple) -> str:
            entry = self._entries.get(candidate)
            return entry.decision if entry is not None else ""

        with self._lock:
            field = field_of(key)
            resident_key = self._resident_key(key)
            if field == "cpu":
                return True, key
            if resident_key != key:
                resident_field = field_of(resident_key)
                if resident_field == "cpu":
                    return True, resident_key
                if field == "cuda":
                    return False, key
                if resident_field == "cuda":
                    return False, resident_key
            return False, key

    def mark_resident_upload_unneeded(self, key: tuple) -> None:
        with self._lock:
            self._resident_upload_unneeded[key] = None
            self._resident_upload_unneeded.move_to_end(key)
            while len(self._resident_upload_unneeded) > self.max_entries:
                self._resident_upload_unneeded.popitem(last=False)

    def resident_upload_unneeded(self, key: tuple) -> bool:
        with self._lock:
            return key in self._resident_upload_unneeded

    def wants_cpu_sample(self, key: tuple) -> bool:
        with self._lock:
            entry = self._entries.get(key)
            return bool(
                entry is not None and not entry.decision and len(entry.cpu_ms) < self.cpu_samples
                and entry.cuda_warmup_done
            )

    def record(self, key: tuple, backend: str, elapsed_sec: float) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = _CrossoverSamples()
                self._entries[key] = entry
                while len(self._entries) > self.max_entries:
                    self._entries.popitem(last=False)
            else:
                self._entries.move_to_end(key)
            if entry.decision:
                return
            milliseconds = float(elapsed_sec) * 1000.0
            if backend == "cuda":
                if not entry.cuda_warmup_done:
                    entry.cuda_warmup_done = True  # includes native plan creation
                elif len(entry.cuda_ms) < self.cuda_samples:
                    entry.cuda_ms.append(milliseconds)
            elif backend == "cpu" and len(entry.cpu_ms) < self.cpu_samples:
                entry.cpu_ms.append(milliseconds)
            if len(entry.cuda_ms) >= self.cuda_samples and len(entry.cpu_ms) >= self.cpu_samples:
                cpu_ms = statistics.median(entry.cpu_ms)
                cuda_ms = statistics.median(entry.cuda_ms)
                entry.decision = "cpu" if cpu_ms * self.margin < cuda_ms else "cuda"

    def report(self, key: tuple) -> dict:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return {"decision": ""}
            return {
                "decision": entry.decision,
                "cpu_median_ms": round(statistics.median(entry.cpu_ms), 4) if entry.cpu_ms else None,
                "cuda_median_ms": round(statistics.median(entry.cuda_ms), 4) if entry.cuda_ms else None,
                "margin": self.margin,
            }
