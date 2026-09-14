from __future__ import annotations

from contextlib import contextmanager
import threading
from pathlib import Path

from core.ai_runtime import AiModelSessionManager
from core.detector_manager import DetectorManager
from core.gpu_runtime import GpuRuntime, GpuRuntimeError
from core.recipe_manager import RecipeManager


class GpuExecutionSession:
    """Own one long-lived runtime/context shared by compatible pipeline runs."""

    def __init__(
        self,
        runtime: GpuRuntime,
        requested: bool,
        config: dict,
        workload: str = "latency",
        ai_session_manager: AiModelSessionManager | None = None,
    ):
        self.runtime = runtime
        self.requested = bool(requested)
        self._dll_path = GpuRuntime._resolve_path(
            str(config.get("dll_path", GpuRuntime.DEFAULT_DLL))
        )
        self._fallback_to_cpu = RecipeManager().gpu_fallback_enabled(config)
        self.workload = workload
        self.ai_session_manager = ai_session_manager or AiModelSessionManager(
            gpu_mode=RecipeManager().gpu_mode(config),
            fallback_to_cpu=RecipeManager().gpu_fallback_enabled(config),
            queue_depth=(
                1 if workload == "latency" else int(config.get("queue_depth", 8))
            ),
        )
        self._closed = False
        self._pipeline_lock = threading.RLock()

    @classmethod
    def from_recipe(cls, recipe: dict, workload: str = "latency") -> "GpuExecutionSession":
        gpu_config = recipe.get("gpu", {}) or {}
        manager = RecipeManager()
        detector_configs = manager.enabled_detectors(recipe)
        requested = manager.gpu_feature_requested(gpu_config, "tiling") or (
            manager.gpu_mode(gpu_config) != "cpu"
            and any(
                bool(config.get("use_gpu", False))
                and DetectorManager.uses_native_cuda_runtime(detector_id)
                for detector_id, config in detector_configs.items()
            )
        )
        runtime = GpuRuntime(
            gpu_config.get("dll_path", GpuRuntime.DEFAULT_DLL),
            fallback_to_cpu=manager.gpu_fallback_enabled(gpu_config),
            enabled=requested,
            queue_depth=(1 if workload == "latency" else int(gpu_config.get("queue_depth", 8))),
            workload=workload,
        )
        return cls(runtime, requested, gpu_config, workload=workload)

    @classmethod
    def from_recipe_path(cls, recipe_path: Path, workload: str = "latency") -> "GpuExecutionSession":
        return cls.from_recipe(RecipeManager().load(Path(recipe_path)), workload=workload)

    def runtime_for(self, gpu_config: dict, requested: bool) -> GpuRuntime:
        if self._closed:
            raise GpuRuntimeError("GPU execution session is already closed")
        requested_path = GpuRuntime._resolve_path(
            str(gpu_config.get("dll_path", GpuRuntime.DEFAULT_DLL))
        )
        fallback_to_cpu = RecipeManager().gpu_fallback_enabled(gpu_config)
        if requested_path != self._dll_path or fallback_to_cpu != self._fallback_to_cpu:
            raise GpuRuntimeError("Injected GPU session is incompatible with the recipe GPU configuration")
        if requested and not self.requested:
            raise GpuRuntimeError("Injected GPU session was created without CUDA enabled")
        # Each pipeline run is a separate recoverable-failure scope on the shared runtime.
        self.runtime.clear_recoverable_error()
        return self.runtime

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.ai_session_manager.close()
        self.runtime.close()

    @contextmanager
    def execution_scope(self):
        if self._closed:
            raise GpuRuntimeError("GPU execution session is already closed")
        with self._pipeline_lock:
            yield

    def __enter__(self) -> "GpuExecutionSession":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class GpuExecutionSessionCache:
    """Lazily reuse one GUI-style session until the recipe identity changes."""

    def __init__(self, workload: str = "latency"):
        self.workload = workload
        self._lock = threading.RLock()
        self._key: tuple[str, int, int] | None = None
        self._session: GpuExecutionSession | None = None

    @staticmethod
    def _recipe_key(recipe_path: Path) -> tuple[str, int, int]:
        resolved = Path(recipe_path).resolve()
        stat = resolved.stat()
        return str(resolved), int(stat.st_mtime_ns), int(stat.st_size)

    def session_for(self, recipe_path: Path) -> GpuExecutionSession:
        key = self._recipe_key(recipe_path)
        with self._lock:
            if self._session is not None and self._key == key:
                return self._session
            self._close_locked()
            session = GpuExecutionSession.from_recipe_path(
                Path(recipe_path), workload=self.workload
            )
            self._session = session
            self._key = key
            return session

    def invalidate(self) -> None:
        with self._lock:
            self._close_locked()

    def close(self) -> None:
        self.invalidate()

    def _close_locked(self) -> None:
        session = self._session
        self._session = None
        self._key = None
        if session is not None:
            session.close()
