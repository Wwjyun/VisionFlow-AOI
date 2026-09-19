"""Test-package bootstrap: keep the suite away from real acquisition hardware.

On the camera machine Sapera LT and `LSI8181_64.dll` are actually installed, so a default
`MainWindow()` in a test would auto-connect the meter-wheel card and write the stored settings into
it. Every test that wants a device passes an explicit `environ` mapping (or the simulator switch),
so the default vendor lookups are pointed at a path that cannot exist. That keeps the suite
deterministic on a development machine, on CI, and on the camera machine alike.

Overriding either variable in the shell has no effect: the isolation is deliberate. A test that
needs a real binding must call the library/loader directly with an explicit path or `environ`.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_ABSENT_HARDWARE_ROOT = Path(tempfile.gettempdir()) / "visionflow-absent-ccd-hardware"

for _variable, _file_name in (
    ("VISIONFLOW_LSI8181_DLL", "LSI8181_64.dll"),
    ("VISIONFLOW_SAPERA_DLL", "DALSA.SaperaLT.SapClassBasic.dll"),
):
    os.environ[_variable] = str(_ABSENT_HARDWARE_ROOT / _file_name)
