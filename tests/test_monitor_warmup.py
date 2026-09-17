from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import DEFAULT, MagicMock, patch

from core.gpu_runtime import GpuRuntimeError
from core.monitor_processor import FolderMonitorProcessor


def _session(requested=True, fallback_to_cpu=True, warm_up=None):
    session = MagicMock()
    session.__enter__.return_value = session
    session.__exit__.return_value = False
    session.requested = requested
    session.fallback_to_cpu = fallback_to_cpu
    if isinstance(warm_up, BaseException):
        session.warm_up.side_effect = warm_up
    else:
        session.warm_up.return_value = dict(warm_up or {"status": "warmed", "image_used": True})
    return session


class MonitorWarmupTests(unittest.TestCase):
    def _run(self, session, warmup_image=True, create_image=True):
        with tempfile.TemporaryDirectory(prefix="visionflow_monitor_warmup_") as temporary:
            root = Path(temporary)
            watch = root / "watch"
            watch.mkdir()
            sample = root / "sample.bmp"
            if create_image:
                sample.write_bytes(b"image")
            events = []
            session.warm_up.side_effect = self._recording(session.warm_up.side_effect, events)
            progress = []
            processor = FolderMonitorProcessor(
                watch,
                root / "recipe.yaml",
                root / "output",
                progress_callback=lambda percent, message: progress.append((percent, message)),
                # The monitor loop checks stop right after warm-up; record the order and stop.
                stop_callback=lambda: events.append("loop") or True,
                warmup_image_path=sample if warmup_image else None,
            )
            with patch(
                "core.monitor_processor.GpuExecutionSession.from_recipe_path", return_value=session
            ) as factory, patch.object(processor, "_process_image") as process_image:
                try:
                    result = processor.run()
                except Exception as exc:
                    return {"error": exc, "events": events, "sample": sample, "factory": factory}
            return {
                "result": result,
                "events": events,
                "sample": sample,
                "factory": factory,
                "process_image": process_image,
                "progress": progress,
            }

    @staticmethod
    def _recording(original, events):
        def warm_up(*args, **kwargs):
            events.append("warm_up")
            if isinstance(original, BaseException):
                raise original
            return DEFAULT

        return warm_up

    def test_monitor_warms_its_own_session_with_the_sample_image_before_watching(self):
        session = _session(warm_up={"status": "warmed", "image_used": True, "pipeline_ms": 140.0})
        run = self._run(session)

        run["factory"].assert_called_once()
        self.assertEqual(run["factory"].call_args.kwargs, {"workload": "throughput"})
        self.assertEqual(run["events"][:2], ["warm_up", "loop"])
        self.assertEqual(session.warm_up.call_count, 1)
        self.assertEqual(session.warm_up.call_args.args[1], run["sample"])
        run["process_image"].assert_not_called()
        summary = run["result"]
        self.assertEqual(summary["processed"], 0)
        self.assertEqual(summary["gpu_warmup"]["status"], "warmed")
        self.assertIn("session_ms", summary["gpu_warmup"])
        self.assertTrue(any(message.startswith("GPU 預熱完成；正在監控") for _, message in run["progress"]))

    def test_missing_or_absent_sample_image_only_creates_the_context(self):
        for warmup_image, create_image in ((False, False), (True, False)):
            with self.subTest(warmup_image=warmup_image):
                session = _session(warm_up={"status": "context_only", "image_used": False})
                run = self._run(session, warmup_image=warmup_image, create_image=create_image)
                self.assertIsNone(session.warm_up.call_args.args[1])
                self.assertEqual(run["result"]["gpu_warmup"]["status"], "context_only")

    def test_auto_mode_keeps_monitoring_when_warm_up_fails_or_cuda_is_unavailable(self):
        failing = _session(fallback_to_cpu=True, warm_up=RuntimeError("anchor not found"))
        run = self._run(failing)
        self.assertEqual(run["events"][:2], ["warm_up", "loop"])
        self.assertEqual(
            (run["result"]["gpu_warmup"]["status"], run["result"]["gpu_warmup"]["reason"]),
            ("failed", "anchor not found"),
        )

        unavailable = _session(fallback_to_cpu=True, warm_up={"status": "unavailable", "reason": "CUDA DLL not found"})
        run = self._run(unavailable)
        self.assertEqual(run["result"]["gpu_warmup"]["status"], "unavailable")
        self.assertTrue(any("CUDA 不可用，改用 CPU" in message for _, message in run["progress"]))

    def test_strict_cuda_fails_before_monitoring_starts(self):
        failing = _session(fallback_to_cpu=False, warm_up=GpuRuntimeError("kernel error"))
        run = self._run(failing)
        self.assertIsInstance(run["error"], GpuRuntimeError)
        self.assertNotIn("loop", run["events"])

        unavailable = _session(fallback_to_cpu=False, warm_up={"status": "unavailable", "reason": "CUDA DLL not found"})
        run = self._run(unavailable)
        self.assertIsInstance(run["error"], GpuRuntimeError)
        self.assertIn("CUDA DLL not found", str(run["error"]))
        self.assertNotIn("loop", run["events"])

    def test_cpu_recipe_is_not_blocked_by_strict_flag(self):
        session = _session(requested=False, fallback_to_cpu=False, warm_up={"status": "not_requested"})
        run = self._run(session)
        self.assertEqual(run["result"]["gpu_warmup"]["status"], "not_requested")
        self.assertIn("loop", run["events"])
        self.assertTrue(any(message.startswith("正在監控") for _, message in run["progress"]))

    def test_gui_monitor_worker_passes_the_loaded_image_as_warm_up_sample(self):
        from gui.workers import FolderMonitorWorker

        worker = FolderMonitorWorker(
            Path("watch"), Path("recipe.yaml"), Path("output"), warmup_image_path=Path("input.bmp")
        )
        with patch("gui.workers.FolderMonitorProcessor") as processor_type:
            processor_type.return_value.run.return_value = {"processed": 0}
            worker.run()
        self.assertEqual(processor_type.call_args.kwargs["warmup_image_path"], Path("input.bmp"))


if __name__ == "__main__":
    unittest.main()
