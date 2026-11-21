"""Compare adjacent DBNet lines in their shared reading direction."""
from math import hypot

import cv2
import numpy as np


def _frame(line, vertical):
    a, b, c, d = line
    if vertical:
        a, b, c, d = a, d, c, b
    dx = (b[0] - a[0] + c[0] - d[0]) / 2
    dy = (b[1] - a[1] + c[1] - d[1]) / 2
    length = hypot(dx, dy)
    width = hypot(d[0] - a[0] + c[0] - b[0], d[1] - a[1] + c[1] - b[1]) / 2
    return (a[0] + c[0]) / 2, (a[1] + c[1]) / 2, dx, dy, length, width


def adjacent_lines(lines, second, vertical, similar_size):
    first = lines[-1]
    x, y, ax, ay, al, aw = _frame(first, vertical)
    x2, y2, bx, by, bl, bw = _frame(second, vertical)
    if min(al, bl, aw, bw) <= 0 or ax * bx + ay * by < .866 * al * bl:
        return False
    ux, uy = ax / al + bx / bl, ay / al + by / bl
    norm = hypot(ux, uy)
    dx, dy = x2 - x, y2 - y
    along = abs(dx * ux + dy * uy) / norm
    across = abs(dx * uy - dy * ux) / norm
    overlap = (al + bl) / 2 - along
    size = (aw + bw) / 2
    if not similar_size:
        # Intersecting ruby or punctuation can be much smaller than the main line.
        if across > size + min(al, bl) / 4:
            return False
        return cv2.intersectConvexConvex(np.asarray(first, np.float32), np.asarray(second, np.float32))[0] > 0
    if across < min(aw, bw) / 2:  # Fragments meeting on the same baseline.
        return overlap >= -1
    end = abs(along - abs(al - bl) / 2)
    if vertical:
        if overlap < .4 * min(al, bl):
            return False
        if end <= min(aw, bw) / 4:
            return across <= 2.1 * size
        offset = min(along, end)
        return (across <= size and offset <= 1.5 * min(aw, bw)
                or hypot(across, offset) <= 1.5 * max(aw, bw) and across <= 1.8 * size)
    if len(lines) > 1 and al > 2 * bl:
        # A wide detected row can straddle two columns. Keep the following short row in its column.
        sx, sy, _, _, start_length, _ = _frame(lines[0], False)
        drift = abs((x2 - sx) * ux + (y2 - sy) * uy) / norm
        if (start_length + bl) / 2 - drift < .3 * min(start_length, bl):
            return False
    # Extra interline leading is common in parallel slanted captions, not at a change of angle.
    tilted = max(abs(ay) / al, abs(by) / bl) > .05 and abs(ax * by - ay * bx) < .1 * al * bl
    gap = .9 if tilted and min(end, along) <= min(aw, bw) / 3 else (.6 if size < 24 else .5)
    return overlap >= .3 * min(al, bl) and across <= size * (1 + gap) + 1


def covered_lines(inner, outer):
    """A paragraph's empty bounding-box corners are not duplicate detections."""
    candidates = [np.asarray(line, np.float32) for line in outer]
    for line in inner:
        quad = np.asarray(line, np.float32)
        area = .9 * cv2.contourArea(quad)
        if not any(cv2.intersectConvexConvex(quad, other)[0] > area for other in candidates):
            return False
    return True
