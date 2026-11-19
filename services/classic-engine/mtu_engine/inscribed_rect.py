"""Exact MTU histogram search, visiting height changes rather than every column."""
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
    heights = np.zeros(width + 2, dtype=np.int32)  # Zero sentinels open and flush runs.
    best_area, best = 0, (0, 0, 0, 0)
    for y, row in enumerate(pixels[y0:y0 + height, x0:x0 + width], y0):
        if (y - y0) % 64 == 0:
            check_cancelled()
        heights[1:-1] += 1
        heights[1:-1][~(row > 0)] = 0
        stack = []
        for x in np.flatnonzero(heights[1:] != heights[:-1]).tolist():
            h, left = int(heights[x + 1]), x
            while stack and stack[-1][0] > h:
                bar, left = stack.pop()
                area = bar * (x - left)
                if area > best_area:
                    best_area, best = area, (x0 + left, y - bar + 1, x - left, bar)
            if h and (not stack or stack[-1][0] < h):
                stack.append((h, left))
    return best
