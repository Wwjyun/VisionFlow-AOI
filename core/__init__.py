
"""Shared runtime settings applied before importing OpenCV."""

import os


# OpenCV reads these limits when its native module is loaded. Keep image-size
# validation from rejecting production images; actual allocation still depends
# on available RAM and OpenCV's integer-sized image dimensions.
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**63 - 1)
os.environ["OPENCV_IO_MAX_IMAGE_WIDTH"] = str(2**31 - 1)
os.environ["OPENCV_IO_MAX_IMAGE_HEIGHT"] = str(2**31 - 1)
