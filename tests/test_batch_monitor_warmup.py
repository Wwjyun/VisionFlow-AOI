from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.batch_processor import BatchImageResult, BatchInspectionProcessor
from core.gpu_runtime import GpuRuntimeError
from core.gpu_session import GpuExecutionSession
from core.monitor_processor import FolderMonitorProcessor


class _Runtime:
    available = True
    unavailable_reason = ""
    device_name = "Fake RTX"

    def clear_recoverable_error(self):
        pass

    def close(self):
        pass


def _real_session(requested=True, fallback_to_cpu=True, available=True):
    runtime = _Runtime()
    runtime.available = available
    runtime.unavailable_reason = "" if available else "CUDA DLL not found"
    config = {"dll_path": "fake.dll", "fallback_to_cpu": fallback_to_cpu}
    if not fallback_to_cpu:
        config["mode"] = "cuda"
    return GpuExecutionSession(runtime, requested=requested, config=config)


def _fake_session(summary=None, error=None, events=None):
    session = MagicMock()
    session.__enter__.return_value = session
    session.__exit__.return_value = False

    def warm_up_before_run(*args, **kwargs):
        if events is not None:
            events.append("warm_up")
        if error is not None:
            raise error
        return dict(summary or {"status": "warmed", "image_used": True})

    session.warm_up_before_run.side_effect = warm_up_before_run
    return session


class SessionWarmUpBeforeRunTests(unittest.TestCase):
    """``gpu.mode`` is enforced once, before a batch or monitor run starts."""

    def test_auto_mode_records_failures_and_unavailable_cuda_without_raising(self):
        session = _real_session(fallback_to_cpu=True)
        with patch.object(session, "warm_up", side_effect=RuntimeError("anchor not found")):
            summary = session.warm_up_before_run(Path("recipe.yaml"), None)
        self.assertEqual((summary["status"], summary["reason"]), ("failed", "anchor not found"))

        unavailable = _real_session(fallback_to_cpu=True, available=False)
        summary = unavailable.warm_up_before_run(Path("recipe.yaml"))
        self.assertEqual((summary["status"], summary["reason"]), ("unavailable", "CUDA DLL not found"))
        self.assertIn("CUDA 不可用，改用 CPU", GpuExecutionSession.warm_up_notice(summary))

    def test_strict_cuda_raises_before_the_run(self):
        session = _real_session(fallback_to_cpu=False)
        with patch.object(session, "warm_up", side_effect=GpuRuntimeError("kernel error")):
            with self.assertRaisesRegex(GpuRuntimeError, "kernel error"):
                session.warm_up_before_run(Path("recipe.yaml"), None)

        unavailable = _real_session(fallback_to_cpu=False, available=False)
        with self.assertRaisesRegex(GpuRuntimeError, "嚴格 CUDA 模式無法開始檢測：CUDA DLL not found"):
            unavailable.warm_up_before_run(Path("recipe.yaml"))

    def test_cpu_recipe_is_not_blocked_by_the_strict_flag(self):
        session = _real_session(requested=False, fallback_to_cpu=False, available=False)
        summary = session.warm_up_before_run(Path("recipe.yaml"))
        self.assertEqual(summary["status"], "not_requested")
        self.assertEqual(GpuExecutionSession.warm_up_notice(summary), "")

    def test_without_image_only_reports_the_context(self):
        session = _real_session()
        summary = session.warm_up_before_run(Path("recipe.yaml"))
        self.assertEqual(summary["status"], "context_only")
        self.assertFalse(summary["image_used"])
        self.assertTrue(GpuExecutionSession.warm_up_notice(summary).startswith("已建立 CUDA context"))

    def test_missing_sample_image_falls_back_to_context_only(self):
        session = _real_session()
        with tempfile.TemporaryDirectory(prefix="visionflow_warmup_missing_") as temporary:
            missing = Path(temporary) / "gone.bmp"
            with patch.object(session, "warm_up", wraps=session.warm_up) as warm_up:
                summary = session.warm_up_before_run(Path("recipe.yaml"), missing)
        self.assertIsNone(warm_up.call_args.args[1])
        self.assertEqual(summary["status"], "context_only")
        self.assertTrue(summary["sample_image_missing"])


class MonitorWarmupTests(unittest.TestCase):
    def _run(self, session, events, warmup_image=True):
        with tempfile.TemporaryDirectory(prefix="visionflow_monitor_warmup_") as temporary:
            root = Path(temporary)
            watch = root / "watch"
            watch.mkdir()
            sample = root / "sample.bmp"
            sample.write_bytes(b"image")
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
                result = processor.run()
        return result, sample, factory, process_image, progress

    def test_monitor_warms_its_own_session_with_the_sample_image_before_watching(self):
        events = []
        session = _fake_session({"status": "warmed", "image_used": True, "pipeline_ms": 140.0}, events=events)
        result, sample, factory, process_image, progress = self._run(session, events)

        self.assertEqual(factory.call_args.kwargs, {"workload": "throughput"})
        self.assertEqual(events[:2], ["warm_up", "loop"])
        self.assertEqual(session.warm_up_before_run.call_count, 1)
        self.assertEqual(session.warm_up_before_run.call_args.args[1], sample)
        process_image.assert_not_called()
        self.assertEqual(result["processed"], 0)
        self.assertEqual(result["gpu_warmup"]["status"], "warmed")
        self.assertIn("session_ms", result["gpu_warmup"])
        self.assertTrue(any(message.startswith("GPU 預熱完成；正在監控") for _, message in progress))

    def test_monitor_without_loaded_image_passes_no_sample(self):
        events = []
        session = _fake_session({"status": "context_only", "image_used": False}, events=events)
        result, _, _, _, _ = self._run(session, events, warmup_image=False)
        self.assertIsNone(session.warm_up_before_run.call_args.args[1])
        self.assertEqual(result["gpu_warmup"]["status"], "context_only")

    def test_strict_failure_stops_the_monitor_before_watching(self):
        events = []
        session = _fake_session(error=GpuRuntimeError("嚴格 CUDA 模式無法開始檢測：no DLL"), events=events)
        with self.assertRaises(GpuRuntimeError):
            self._run(session, events)
        self.assertNotIn("loop", events)

    def test_gui_monitor_worker_passes_the_loaded_image_as_warm_up_sample(self):
        from gui.workers import FolderMonitorWorker

        worker = FolderMonitorWorker(
            Path("watch"), Path("recipe.yaml"), Path("output"), warmup_image_path=Path("input.bmp")
        )
        with patch("gui.workers.FolderMonitorProcessor") as processor_type:
            processor_type.return_value.run.return_value = {"processed": 0}
            worker.run()
        self.assertEqual(processor_type.call_args.kwargs["warmup_image_path"], Path("input.bmp"))


class BatchWarmupTests(unittest.TestCase):
    def _run(self, session, events, images=2):
        with tempfile.TemporaryDirectory(prefix="visionflow_batch_warmup_") as temporary:
            root = Path(temporary)
            paths = [root / f"image{index}.png" for index in range(images)]
            progress = []
            processor = BatchInspectionProcessor(
                root,
                root / "recipe.yaml",
                root / "output",
                max_workers=1,
                progress_callback=lambda percent, message: progress.append((percent, message)),
            )
            processor.discover_images = lambda: list(paths)

            def process_image(image_path, _output_dir, gpu_session):
                events.append(("image", image_path.name, gpu_session))
                return BatchImageResult(image_path, "PASS", 0, 0, 1, 0.01, {}, {})

            with patch(
                "core.batch_processor.GpuExecutionSession.from_recipe_path", return_value=session
            ) as factory, patch.object(processor, "_process_image", side_effect=process_image), patch(
                "core.batch_processor.CsvSummaryExporter.write_summary", return_value=None
            ):
                result = processor.run()
        return result, factory, progress

    def test_batch_prepares_the_context_once_without_a_sample_run_before_any_image(self):
        events = []
        session = _fake_session({"status": "context_only", "image_used": False}, events=events)
        result, factory, progress = self._run(session, events)

        self.assertEqual(factory.call_args.kwargs, {"workload": "throughput"})
        self.assertEqual(session.warm_up_before_run.call_count, 1)
        self.assertEqual(len(session.warm_up_before_run.call_args.args), 1)
        self.assertNotIn("progress_callback", session.warm_up_before_run.call_args.kwargs)
        self.assertEqual(events[0], "warm_up")
        self.assertEqual([event[1] for event in events[1:]], ["image0.png", "image1.png"])
        self.assertTrue(all(event[2] is session for event in events[1:]))
        self.assertEqual(result["summary"]["total"], 2)
        self.assertEqual([item["image_name"] for item in result["items"]], ["image0.png", "image1.png"])
        self.assertEqual(result["gpu_warmup"]["status"], "context_only")
        self.assertIn("session_ms", result["gpu_warmup"])
        self.assertTrue(any(
            message.startswith("已建立 CUDA context（第一張仍需配置裝置記憶體）；批量檢測執行中")
            for _, message in progress
        ))

    def test_auto_unavailable_keeps_processing_every_image(self):
        events = []
        session = _fake_session({"status": "unavailable", "reason": "CUDA DLL not found"}, events=events)
        result, _, progress = self._run(session, events)
        self.assertEqual(result["summary"]["total"], 2)
        self.assertEqual(result["gpu_warmup"]["status"], "unavailable")
        self.assertTrue(any("CUDA 不可用，改用 CPU" in message for _, message in progress))

    def test_strict_failure_stops_the_batch_before_any_image(self):
        events = []
        session = _fake_session(error=GpuRuntimeError("嚴格 CUDA 模式無法開始檢測：no DLL"), events=events)
        with self.assertRaises(GpuRuntimeError):
            self._run(session, events)
        self.assertEqual(events, ["warm_up"])

    def test_empty_folder_does_not_create_a_session(self):
        events = []
        session = _fake_session(events=events)
        result, factory, _ = self._run(session, events, images=0)
        factory.assert_not_called()
        self.assertNotIn("gpu_warmup", result)


if __name__ == "__main__":
    unittest.main()
