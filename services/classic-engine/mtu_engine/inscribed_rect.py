"""Exact MTU histogram search over equal rows and runs of equal height."""
import cv2
import numpy as np

from .backend import check_cancelled


def largest_rectangle(mask, bounds=None):
    pixels = np.asarray(mask)
    if pixels.ndim != 2 or not pixels.size:
        return 0, 0, 0, 0
    if bounds is None:
        bounds = cv2.boundingRect(pixels if pixels.dtype == np.uint8 else (pixels > 0).view(np.uint8))
    x0, y0, width, height = bounds
    roi = pixels[y0:y0 + height, x0:x0 + width]
    heights = np.zeros(roi.shape[1] + 2, dtype=np.int32)  # Zero sentinels open and flush runs.
    best_area, best = 0, (0, 0, 0, 0)
    for y in range(0, len(roi), 64):
        check_cancelled()
        rows = roi[y:y + 64] > 0
        ends = np.flatnonzero(np.any(rows[1:] != rows[:-1], axis=1)) + 1
        start = 0
        # The last row of an identical run dominates every earlier rectangle in it.
        for end in [*ends.tolist(), len(rows)]:
            heights[1:-1] += end - start
            heights[1:-1][~rows[start]] = 0
            stack = []
            for x in np.flatnonzero(heights[1:] != heights[:-1]).tolist():
                h, left = int(heights[x + 1]), x
                while stack and stack[-1][0] > h:
                    bar, left = stack.pop()
                    area = bar * (x - left)
                    if area > best_area:
                        best_area, best = area, (x0 + left, y0 + y + end - bar, x - left, bar)
                if h and (not stack or stack[-1][0] < h):
                    stack.append((h, left))
            start = end
    return best
