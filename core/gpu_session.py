from __future__ import annotations

from contextlib import contextmanager
import tempfile
import threading
import time
from pathlib import Path

from core.ai_runtime import AiModelSessionManager
from core.detector_manager import DetectorManager
from core.gpu_runtime import GpuRuntime, GpuRuntimeError
from core.host_image_buffers import HostImageBufferPool
from core.logging_system import LogMixin
from core.recipe_manager import RecipeManager


class GpuExecutionSession(LogMixin):
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
        # Runs on this session are serialized, so one reusable decoded-image backing serves them all.
        self.host_image_buffers = HostImageBufferPool(runtime)

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
        # Unpin the host backing while the CUDA context that registered it still exists.
        self.host_image_buffers.close()
        self.runtime.close()

    # Every artifact a pipeline run can write; a warm-up must leave nothing on disk.
    WARM_UP_OUTPUT_OVERRIDES = {
        "save_overlay": False,
        "save_ng_tiles": False,
        "save_csv": False,
        "save_matrix_csv": False,
        "save_json": False,
        "save_debug_images": False,
    }

    def warm_up(self, recipe_path: Path, image_path: Path | None = None, progress_callback=None) -> dict:
        """Pay the first-inspection cost on this session before the first real image.

        The session already holds the DLL and CUDA context. With an image, the real pipeline runs
        once through this session with every output disabled, so the resident upload and detector
        buffers are allocated at the production image size and the next inspection starts warm.
        The result is discarded: a warm-up is not an inspection and never writes overlays, CSV,
        JSON, NG tiles or debug images.

        RTX 3090 measurement (16384x13000, six 12000x2000 ROIs, ``202-CS-SN-1``): session creation
        ~107 ms and a first run ~140 ms slower than later runs, with every device buffer allocated
        during that first run.
        """
        from core.pipeline import AOIPipeline

        def report(percent: int, message: str) -> None:
            if progress_callback is not None:
                progress_callback(int(percent), message)

        runtime = self.runtime
        summary = {
            "status": "",
            "pipeline_ms": 0.0,
            "image_used": False,
            "device_name": "",
            "reason": "",
        }
        if not self.requested:
            summary.update(status="not_requested", reason="此 Recipe 未啟用 CUDA，不需要預熱")
            return summary
        if not getattr(runtime, "available", False):
            summary.update(
                status="unavailable",
                reason=str(getattr(runtime, "unavailable_reason", "") or "CUDA 不可用"),
            )
            return summary
        summary["device_name"] = str(getattr(runtime, "device_name", "") or "")
        if image_path is None:
            summary.update(status="context_only", reason="未載入影像，只建立 CUDA context")
            summary.update(self._context_summary(runtime))
            return summary

        report(20, "正在以目前影像試跑（不輸出檔案）")
        with tempfile.TemporaryDirectory(prefix="visionflow_gpu_warmup_") as temporary:
            pipeline = AOIPipeline(
                Path(recipe_path),
                Path(temporary),
                progress_callback=lambda percent, _message: report(20 + int(percent) * 3 // 4, "GPU 預熱試跑中"),
                output_overrides=dict(self.WARM_UP_OUTPUT_OVERRIDES),
                gpu_session=self,
            )
            started = time.perf_counter()
            result = pipeline.run(Path(image_path))
            summary["pipeline_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        gpu = (result.get("execution", {}) or {}).get("gpu", {}) or {}
        detectors = gpu.get("detectors", {}) or {}
        fallback_reasons = [
            str(status.get("fallback_reason") or "")
            for status in detectors.values()
            if status.get("requested") and status.get("fallback_reason")
        ]
        summary["image_used"] = True
        summary["resident_upload"] = bool((gpu.get("resident_image", {}) or {}).get("active", False))
        summary["device_host_split"] = dict(gpu.get("device_host_split", {}) or {})
        summary.update(self._context_summary(runtime))
        if fallback_reasons:
            summary.update(status="fallback", reason=fallback_reasons[0])
        else:
            summary["status"] = "warmed"
        report(100, "GPU 預熱完成")
        return summary

    def warm_up_before_run(self, recipe_path: Path, image_path: Path | None = None, progress_callback=None) -> dict:
        """Prepare this session before a batch or monitor run and enforce ``gpu.mode`` up front.

        Batch passes no image: a sample run costs more than the one-time cold start it would save,
        and the DLL/CUDA context already exists once the session is built, so only its state is
        recorded. Monitor passes the operator's loaded image because it waits for the first product
        anyway. ``gpu.mode: cuda`` raises before the run when CUDA is unavailable or the warm-up
        fails, instead of turning every image into an ERROR; ``auto`` records the reason and the
        run continues on the normal per-image fallback path.
        """
        sample_missing = image_path is not None and not Path(image_path).is_file()
        if sample_missing:
            self.logger.warning("GPU warm-up sample image is missing, using context only: %s", image_path)
            image_path = None
        strict = self.requested and not self._fallback_to_cpu
        try:
            summary = self.warm_up(Path(recipe_path), image_path, progress_callback=progress_callback)
        except Exception as exc:
            if strict:
                raise
            self.logger.warning("GPU warm-up failed, the run continues: %s", exc, exc_info=True)
            summary = {"status": "failed", "image_used": image_path is not None, "reason": str(exc)}
        if sample_missing:
            summary["sample_image_missing"] = True
        if strict and summary.get("status") == "unavailable":
            raise GpuRuntimeError(f"嚴格 CUDA 模式無法開始檢測：{summary.get('reason', '')}")
        return summary

    @staticmethod
    def warm_up_notice(summary: dict) -> str:
        """Short operator-facing prefix for batch/monitor progress text; empty for CPU recipes."""
        status = summary.get("status", "")
        reason = str(summary.get("reason", "") or "")
        if status == "warmed":
            return "GPU 預熱完成；"
        if status == "context_only":
            return "已建立 CUDA context（第一張仍需配置裝置記憶體）；"
        if status == "fallback":
            return f"GPU 預熱時 Detector 改用 CPU：{reason}；"
        if status == "unavailable":
            return f"CUDA 不可用，改用 CPU：{reason}；"
        if status == "failed":
            return f"GPU 預熱失敗，繼續執行：{reason}；"
        return ""

    @staticmethod
    def _context_summary(runtime) -> dict:
        stats = getattr(runtime, "performance_stats", None)
        context = {}
        if callable(stats):
            try:
                context = stats().get("persistent_context", {}) or {}
            except Exception:  # metrics are informative only and must not fail a warm-up
                context = {}
        return {
            "context_active": bool(context.get("active", False)),
            "reserved_bytes": int(context.get("reserved_bytes", 0) or 0),
            "allocation_count": int(context.get("allocation_count", 0) or 0),
        }

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

    def warm_up(self, recipe_path: Path, image_path: Path | None = None, progress_callback=None) -> dict:
        """Create (or reuse) the session for ``recipe_path`` and warm it; see ``GpuExecutionSession.warm_up``."""
        if progress_callback is not None:
            progress_callback(0, "正在建立 GPU session")
        started = time.perf_counter()
        session = self.session_for(Path(recipe_path))
        session_ms = round((time.perf_counter() - started) * 1000.0, 1)
        summary = session.warm_up(Path(recipe_path), image_path, progress_callback=progress_callback)
        return {"session_ms": session_ms, **summary}

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
