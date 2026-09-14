"""OpenCV reference implementation of connected components with stats.

Verified equivalent to `cv2.connectedComponentsWithStats` for both 4- and 8-connectivity: the label
map, the label numbering and order, the per-label stats and the centroids all match. This matters
because the 202-CS-SN-1 detector walks components in label order, so its candidate ordering - and
therefore its defect list - depends on that order.

Only the two connectivities and the 8-bit single-channel input the detectors use are covered.
"""

from __future__ import annotations

import numpy as np


def connected_components_with_stats(mask: np.ndarray, connectivity: int = 8):
    """Return ``(count, labels, stats, centroids)`` exactly as OpenCV does.

    ``count`` includes the background label 0. Label 0's stats describe the background pixels;
    when a label has no pixels, its stats are the sentinel ``(-1, INT_MAX, 0, 0, 0)`` and its
    centroid is NaN, matching OpenCV.
    """
    height, width = mask.shape[:2]
    foreground = mask > 0
    labels = np.zeros((height, width), dtype=np.int32)
    parent: list[int] = [0]

    def find(label: int) -> int:
        root = label
        while parent[root] != root:
            root = parent[root]
        while parent[label] != root:
            parent[label], label = root, parent[label]
        return root

    def union(first: int, second: int) -> int:
        root_a, root_b = find(first), find(second)
        if root_a == root_b:
            return root_a
        if root_a < root_b:
            parent[root_b] = root_a
            return root_a
        parent[root_a] = root_b
        return root_b

    neighbours = (
        ((0, -1), (-1, -1), (-1, 0), (-1, 1)) if connectivity == 8 else ((0, -1), (-1, 0))
    )

    appearance: list[int] = []
    for y in range(height):
        row = foreground[y]
        for x in range(width):
            if not row[x]:
                continue
            found = []
            for dy, dx in neighbours:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and labels[ny, nx]:
                    found.append(labels[ny, nx])
            if not found:
                parent.append(len(parent))
                labels[y, x] = len(parent) - 1
                appearance.append(labels[y, x])
            else:
                root = found[0]
                for other in found[1:]:
                    root = union(root, other)
                labels[y, x] = find(root)

    # OpenCV numbers components in the order their provisional label first appears in the raster
    # scan, so replay that order over the discovery list and map every root to its position.
    ordered_roots: list[int] = []
    for provisional in appearance:
        root = find(provisional)
        if root not in ordered_roots:
            ordered_roots.append(root)
    canonical = np.zeros_like(labels)
    for y in range(height):
        for x in range(width):
            if labels[y, x]:
                canonical[y, x] = ordered_roots.index(find(labels[y, x])) + 1

    count = len(ordered_roots) + 1
    stats = np.zeros((count, 5), dtype=np.int32)
    centroids = np.zeros((count, 2), dtype=np.float64)
    for label in range(count):
        ys, xs = np.nonzero(canonical == label)
        if xs.size == 0:
            stats[label] = (-1, np.iinfo(np.int32).max, 0, 0, 0)
            centroids[label] = (np.nan, np.nan)
            continue
        stats[label] = (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1, xs.size)
        centroids[label] = (float(xs.mean()), float(ys.mean()))
    return count, canonical, stats, centroids
