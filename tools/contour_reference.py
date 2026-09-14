"""Pure-Python/NumPy reference implementation of ``cv2.findContours``.

This module is a faithful port of the border-following algorithm that OpenCV runs for
``cv2.findContours(image, cv2.RETR_LIST | cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)``:

* Suzuki, S. and Abe, K., "Topological Structural Analysis of Digitized Binary Images by
  Border Following", CVGIP 30(1), pp. 32-46, 1985 -- the outer/hole border labelling rule.
* OpenCV ``modules/imgproc/src/contours.cpp``:
  - ``cvStartFindContours_Impl``  -- 1-pixel zero frame, binarization, scanner state.
  - ``cvFindNextContour``         -- raster scan, outer/hole classification, ``lnbd``.
  - ``icvFetchContour``           -- the actual 8-neighbour border trace (``CV_CHAIN_APPROX_SIMPLE``).

The reference exists so that a future GPU operator (see ``AGENT.md``, "CPU/GPU architecture
contract") can be checked against the exact CPU behaviour -- including contour *order* and
*point order* -- without calling OpenCV.

Scope and limits
----------------
Only ``mode="list"`` (``cv2.RETR_LIST``) and ``mode="external"`` (``cv2.RETR_EXTERNAL``) with
``CHAIN_APPROX_SIMPLE`` are supported. ``RETR_CCOMP`` / ``RETR_TREE`` / ``RETR_FLOODFILL`` take
the ``icvFetchContourEx`` path with the ``cinfo_table``/``icvTraceContour`` parent search and are
deliberately not implemented here. ``CHAIN_APPROX_NONE``/``CHAIN_CODE``/TC89 are also out of scope.

The image is treated exactly like OpenCV does: a 1-pixel zero border is added, and the padded
array is binarized with ``THRESH_BINARY`` (threshold 0), so any non-zero input pixel becomes 1.
Returned coordinates are 0-based pixel coordinates in the original image (OpenCV's
``offset = (-1, -1)`` compensating for the added frame).

Bookkeeping notes for review against ``icvFetchContour``
-------------------------------------------------------
8-neighbour ring order (``CV_INIT_3X3_DELTAS`` with ``step`` = one padded row, y growing down)::

    index   :  0     1      2      3      4      5      6     7
    dir     :  E     NE     N      NW     W      SW     S     SE
    (dy,dx) : (0,1) (-1,1) (-1,0) (-1,-1) (0,-1) (1,-1) (1,0) (1,1)

``deltas`` has 16 entries in C: indices 8..15 mirror 0..7. The search loop writes to
``deltas[++s]`` and only stops at ``s == MAX_SIZE - 1 == 15``, so the mirroring lets the ring
rotation be expressed as a simple upward index scan without a modulo inside the search loop.
This port keeps the 16-entry ``_DELTAS`` table and the same ``while s < 15`` bound so that an
exhausted search leaves ``s == 15 -> s & 7 == 7`` exactly as C does.

* Start of the trace: ``s_end = s = 0`` for a hole border, ``4`` for an outer border; the scan
  then walks ``s = (s - 1) & 7`` (NW,N,NE,E,SE,S,SW,W for outer; SE,S,SW,W,NW,N,NE,E for hole)
  until a non-zero neighbour is found or ``s`` comes back to ``s_end``. ``s == s_end`` means a
  single-pixel domain: the pixel gets marked ``nbd | -128`` and one point is emitted.
* ``i1`` is the neighbour found by that initial scan; ``i0`` is the border pixel. The trace ends
  when the *edge* repeats: ``i4 == i0 && i3 == i1`` -- i.e. the pixel we move to is the start
  pixel and the pixel we moved from is the pixel the trace originally left ``i0`` towards. In the
  original Suzuki/Abe formulation this is the ``delx``/``dely``/``delayat`` restart bookkeeping:
  OpenCV folds that bookkeeping into the ``(i0, i1)`` start edge plus the per-step
  ``s_end``/``prev_s`` state, and the trace is finished exactly when the start edge is
  re-traversed.
* Per step the incoming direction is ``s_end`` (the direction from ``i3`` back to the previous
  pixel). The next neighbour is searched starting at ``s_end + 1``, i.e. "previous point index + 1"
  rotating in the E->NE->... direction. After moving, the new incoming direction is ``s = (s + 4) & 7``
  (reverse of the step just taken).
* ``prev_s`` is initialised to ``s ^ 4`` (the *reverse* of the entry direction) rather than to
  ``s``, which suppresses the first point emission when the very first step continues in the same
  direction -- that is how OpenCV avoids a duplicate point at the start of a straight run.
* Right-bound marking: ``if ((unsigned)(s - 1) < (unsigned)s_end) *i3 = nbd | -128; else if
  (*i3 == 1) *i3 = nbd;`` -- the unsigned cast makes ``s == 0`` false, so the test is
  ``1 <= s <= s_end``. This is what distinguishes a "right turn" pixel (marked with bit 7 set,
  i.e. ``-126``) from a plain outer-border pixel (marked ``2``), and it is why a second border is
  never started at the same pixel: the raster scan only starts a new border where a *0 -> non-zero*
  (outer) or *non-zero -> 0* (hole) transition against an unmarked run occurs.
* ``nbd`` stays ``2`` for ``RETR_LIST``/``RETR_EXTERNAL`` because ``cvFindNextContour`` only
  advances it inside the ``mode > 1`` (``icvFetchContourEx``) branch.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

__all__ = ["find_contours"]

_MODES = {"list": 1, "external": 0}

# CV_INIT_3X3_DELTAS(deltas, step, 1) expressed as (dy, dx) pairs; index order matters.
_RING: Tuple[Tuple[int, int], ...] = (
    (0, 1),     # 0: E
    (-1, 1),    # 1: NE
    (-1, 0),    # 2: N
    (-1, -1),   # 3: NW
    (0, -1),    # 4: W
    (1, -1),    # 5: SW
    (1, 0),     # 6: S
    (1, 1),     # 7: SE
)
# memcpy(deltas + 8, deltas, 8 * sizeof(deltas[0]))
_DELTAS: Tuple[Tuple[int, int], ...] = _RING * 2

_NBD = 2                       # const schar nbd = 2 inside icvFetchContour
_MARKED = -126                 # (schar)(nbd | -128)
_MARK_MASK = -2                # new_mask used by the 8-bit scan ("prev & -2")


def _fetch_contour(
    img: np.ndarray,
    i0_y: int,
    i0_x: int,
    is_hole: int,
    pt_x: int,
    pt_y: int,
) -> List[Tuple[int, int]]:
    """Port of ``icvFetchContour(ptr, step, pt, contour, CV_CHAIN_APPROX_SIMPLE)``.

    ``img`` is the *padded* label image and is mutated in place (marks 2 / -126) exactly like
    OpenCV does; ``(i0_y, i0_x)`` is the padded position of the border pixel; ``(pt_x, pt_y)`` is
    the already-offset start point that OpenCV stores in the sequence.
    """
    points: List[Tuple[int, int]] = []

    # ---- initial search for the neighbour that the trace leaves i0 towards -------------
    s_end = 0 if is_hole else 4
    s = s_end
    while True:
        s = (s - 1) & 7
        dy, dx = _RING[s]
        i1_y, i1_x = i0_y + dy, i0_x + dx
        if img[i1_y, i1_x] != 0:
            break
        if s == s_end:
            break

    if s == s_end:
        # single pixel domain
        img[i0_y, i0_x] = _MARKED
        points.append((pt_x, pt_y))
        return points

    i3_y, i3_x = i0_y, i0_x
    prev_s = s ^ 4
    i4_y, i4_x = 0, 0

    # ---- follow border ------------------------------------------------------------------
    while True:
        s_end = s
        # `s` is always in 0..7 here, so C's `s = min(s, MAX_SIZE - 1)` is a no-op.
        while s < 15:
            s += 1
            dy, dx = _DELTAS[s]
            i4_y, i4_x = i3_y + dy, i3_x + dx
            if img[i4_y, i4_x] != 0:
                break
        s &= 7

        # check "right" bound:  (unsigned)(s - 1) < (unsigned)s_end
        if s >= 1 and (s - 1) < s_end:
            img[i3_y, i3_x] = _MARKED
        elif img[i3_y, i3_x] == 1:
            img[i3_y, i3_x] = _NBD

        # CHAIN_APPROX_SIMPLE: keep a point only where the step direction changes.
        if s != prev_s:
            points.append((pt_x, pt_y))
            prev_s = s

        dy, dx = _RING[s]
        pt_y += dy
        pt_x += dx

        if i4_y == i0_y and i4_x == i0_x and i3_y == i1_y and i3_x == i1_x:
            break

        i3_y, i3_x = i4_y, i4_x
        s = (s + 4) & 7

    return points


def _scan(img: np.ndarray, mode_code: int, width: int, height: int) -> List[np.ndarray]:
    """Port of the ``cvStartFindContours_Impl`` / ``cvFindNextContour`` raster scan."""
    scan_w = width + 1          # scanner->img_size.width  = W + 2 - 1
    scan_h = height + 1         # scanner->img_size.height = H + 2 - 1

    contours: List[np.ndarray] = []
    x, y = 1, 1                 # scanner->pt
    lnbd_x, lnbd_y = 0, 1       # scanner->lnbd
    prev = int(img[y, x - 1])   # int prev = img[x - 1]

    while y < scan_h:
        row = img[y]
        restarted = False
        while x < scan_w:
            # for( ; x < width && (p = img[x]) == prev; x++ ) ;
            while x < scan_w and int(row[x]) == prev:
                x += 1
            if x >= scan_w:
                break
            p = int(row[x])

            is_hole = 0
            # /* if not external contour */
            if not (prev == 0 and p == 1):
                # /* check hole */  (p == 0 and prev >= 1; -126 is < 1)
                if p != 0 or prev < 1:
                    prev = p
                    if prev & _MARK_MASK:       # /* update lnbd */
                        lnbd_x = x
                    x += 1
                    continue
                # `lnbd.x = x - 1` (line 1130) is dead for modes 0/1: line 1197 overwrites it,
                # and for mode 0 the `is_hole` short-circuit resumes the scan anyway.
                is_hole = 1

            # if( mode == 0 && (is_hole || img0[lnbd.y * step + lnbd.x] > 0) ) goto resume_scan;
            if mode_code == 0 and (is_hole or int(img[lnbd_y, lnbd_x]) > 0):
                prev = p
                if prev & _MARK_MASK:
                    lnbd_x = x
                x += 1
                continue

            origin_y = y
            origin_x = x - is_hole
            lnbd_x, lnbd_y = x - is_hole, y       # lnbd.x = x - is_hole

            points = _fetch_contour(img, origin_y, origin_x, is_hole, origin_x - 1, origin_y - 1)
            if points:
                contours.append(np.asarray(points, dtype=np.int32).reshape(-1, 1, 2))

            # scanner->pt.x = x + 1; scanner->pt.y = y; (8-bit path)
            x += 1
            prev = int(img[y, x - 1])
            restarted = True
            break

        if restarted:
            continue

        lnbd_x, lnbd_y = 0, y + 1
        x = 1
        prev = 0
        y += 1

    return contours


def find_contours(binary: np.ndarray, mode: str) -> List[np.ndarray]:
    """Reproduce ``cv2.findContours(binary, mode, cv2.CHAIN_APPROX_SIMPLE)``.

    Parameters
    ----------
    binary:
        2-D array; every non-zero element is foreground (matching OpenCV's ``THRESH_BINARY``
        binarization of the padded image).
    mode:
        ``"list"`` for ``cv2.RETR_LIST`` or ``"external"`` for ``cv2.RETR_EXTERNAL``.

    Returns
    -------
    list of ``int32`` arrays shaped ``(N, 1, 2)`` in OpenCV's contour order, with
    ``CHAIN_APPROX_SIMPLE`` compression applied and 0-based original-image coordinates.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {sorted(_MODES)}, got {mode!r}")

    src = np.asarray(binary)
    if src.ndim != 2:
        raise ValueError(f"binary must be a 2-D array, got shape {src.shape}")

    height, width = int(src.shape[0]), int(src.shape[1])
    if height == 0 or width == 0:
        return []

    # copyMakeBorder(image0, image, 1, 1, 1, 1, BORDER_CONSTANT | BORDER_ISOLATED, Scalar(0))
    img = np.zeros((height + 2, width + 2), dtype=np.int16)
    # cvThreshold(mat, mat, 0, 1, THRESH_BINARY)
    img[1 : height + 1, 1 : width + 1] = np.where(src != 0, 1, 0)

    contours = _scan(img, _MODES[mode], width, height)

    # Ordering: cvEndFindContours returns `scanner->frame.v_next`, and icvEndProcessContour ->
    # cvInsertNodeIntoTree *prepends* every finished contour to that v_next chain. For
    # RETR_LIST / RETR_EXTERNAL every contour has the image frame as parent, so the chain is
    # [last discovered, ..., first discovered]; cvTreeToNodeSeq then walks h_next from the head.
    # The net effect is that OpenCV reports the flat list in reverse discovery order.
    contours.reverse()
    return contours
