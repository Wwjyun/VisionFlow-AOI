"""Full-resolution traditional-CV detector tuning tool."""

import os


os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**63 - 1)
os.environ["OPENCV_IO_MAX_IMAGE_WIDTH"] = str(2**31 - 1)
os.environ["OPENCV_IO_MAX_IMAGE_HEIGHT"] = str(2**31 - 1)

from .engine import ContourProcessingEngine, ProcessingRecipe, ProcessingResult
from .version import __version__

__all__ = [
    "ContourProcessingEngine",
    "ProcessingRecipe",
    "ProcessingResult",
    "__version__",
]
