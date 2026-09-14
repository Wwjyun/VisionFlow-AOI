"""OpenCV reference implementation of connected components with stats.

``connectivity=4`` is **fully equivalent** to ``cv2.connectedComponentsWithStats``: the label map,
the label numbering, the per-label stats and the centroids all match, including on random masks.

``connectivity=8`` is equivalent in the component count, the pixel sets and the per-component
stats, but **not in the label numbering**: OpenCV labels 8-connectivity with the Bolelli 2x2 block
scan, which creates provisional labels per block corner instead of per pixel.  Until that scan is
reproduced, this reference must not be used as the golden standard for an 8-connectivity
replacement.

The numbering rule itself is reproduced faithfully and documented in ``_numbering`` below.

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

    # OpenCV's ``flattenL`` walks the union-find array ``P`` in ascending *label index*
    # order and hands out consecutive numbers to the roots it meets:
    #
    #     k = 1
    #     for i in 1..lunique-1:
    #         if P[i] < i: P[i] = P[P[i]]   # non-root: adopt the root's number
    #         else:        P[i] = k; k += 1 # root: take the next number
    #
    # ``set_union`` always keeps the smaller provisional label as the root, so a
    # component's root is the smallest provisional label it ever received.  The final
    # number of a component is therefore the rank of that smallest provisional label
    # among all components - *not* the order in which the component first appeared in
    # the raster scan.  The two orders coincide only when no merge ever lowers a
    # component's root, which is why structured masks agreed and random masks did not.
    roots = sorted({find(provisional) for provisional in appearance})
    rank_of_root = {root: index + 1 for index, root in enumerate(roots)}
    canonical = np.zeros_like(labels)
    for y in range(height):
        for x in range(width):
            if labels[y, x]:
                canonical[y, x] = rank_of_root[find(labels[y, x])]

    count = len(roots) + 1
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
