from __future__ import annotations

import os
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui import workers
from gui.workflow_controllers import BatchWorkflowController


class WorkerSignalPayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_result_signals_pass_python_objects_across_threads_without_conversion(self):
        items = [{"image_name": f"img_{index}.bmp", "final_result": "NG", "detail": {"defects": list(range(200))}}
                 for index in range(1000)]
        summary = {"summary": {"total": 1000}, "items": items}
        worker = workers.BatchInspectionWorker(Path("images"), Path("recipe.yaml"), Path("out"))
        received = []
        controller = BatchWorkflowController(None)
        with patch("gui.workers.BatchInspectionProcessor") as processor_type:
            processor_type.return_value.run.return_value = summary
            controller.start(
                worker,
                signal_handlers=((worker.finished, received.append),),
                terminal_signals=(worker.finished,),
                on_thread_finished=lambda: None,
            )
            deadline = time.perf_counter() + 10
            # thread.quit is a queued call on the GUI thread, so keep the event loop running until it stops.
            while (not received or controller.thread.isRunning()) and time.perf_counter() < deadline:
                self.app.processEvents()
                time.sleep(0.005)
            self.assertFalse(controller.thread.isRunning())
        self.assertEqual(len(received), 1)
        self.assertIs(received[0], summary, "the queued signal must not convert the summary to a QVariantMap")

    def test_every_worker_result_signal_is_declared_as_object(self):
        source = Path(workers.__file__).read_text(encoding="utf-8")
        self.assertNotIn("Signal(dict)", source)
        self.assertNotIn("Signal(list)", source)


if __name__ == "__main__":
    unittest.main()
