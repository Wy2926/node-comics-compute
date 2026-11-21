"""Hayai paragraph crop bounds."""
from classic_node.protocol import MAX_REGIONS


def paragraph_box(box, width, height):
    left, top, right, bottom = map(int, box)
    return max(0, left), max(0, top), min(width, right), min(height, bottom)


def line_context_boxes(boxes, blocks, indices, width, height):
    """Recover word spacing and clipped edge glyphs on isolated single lines."""
    if len(indices) > MAX_REGIONS:
        return boxes  # Bound neighbor checks before the final region count is known.
    crops = boxes.copy()
    for i in indices:
        l, t, r, b = boxes[i]
        w, h = r - l, b - t
        if len(blocks[i].lines) != 1:
            continue
        if w >= 2 * h:
            dx = dy = 16
        elif 2 * w <= h <= 4 * w:  # A short vertical aside may start with a partially detected glyph.
            dx, dy = min(16, (w + 1) // 2), min(32, w)
        else:
            continue
        left, top, right, bottom = paragraph_box((l - dx, t - dy, r + dx, b + dy), width, height)
        for j in indices:
            x1, y1, x2, y2 = boxes[j]
            if j != i and x1 < right and left < x2 and y1 < bottom and top < y2:
                break
        else:
            crops[i] = left, top, right, bottom
    return crops
