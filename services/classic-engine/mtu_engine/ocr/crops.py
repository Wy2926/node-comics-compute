"""Hayai paragraph crop bounds."""
import numpy as np
from PIL import Image, ImageDraw

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


def _attached_rows(rows, paper):
    # Preserve a clipped glyph up to the first paper-colored scanline, not the next row of text.
    occupied = np.any(np.abs(rows.astype(np.int16) - paper) > 24, axis=(1, 2))
    separator = np.flatnonzero(~occupied)
    return max(0, (int(separator[0]) if len(separator) else len(rows)) - 1)


def _paper(pixels):
    if pixels.shape[0] > pixels.shape[1]:
        pixels = pixels.swapaxes(0, 1)
    edges = pixels[[0, -1], ::max(1, (pixels.shape[1] + 127) // 128)]
    return np.median(edges, axis=(0, 1)).astype(np.int16)


def paragraph_crop(image, i, boxes, blocks, indices):
    l, t, r, b = boxes[i]
    width, height = image.shape[1], image.shape[0]
    left, top, right, bottom = paragraph_box(blocks[i].xyxy, width, height)
    paper = None
    if (len(blocks[i].lines) == 1 and right - left >= 2 * (bottom - top)
            and (t < top or bottom < b)):
        # Hayai needs vertical whitespace to preserve word spaces, not pixels from the next line.
        xs = slice(left, right, max(1, (right - left + 127) // 128))
        paper = _paper(image[top:bottom, left:right])
        top -= _attached_rows(image[t:top + 1, xs][::-1], paper)
        bottom += _attached_rows(image[bottom - 1:b, xs], paper)
        crop = Image.new('RGB', (r - l, b - t), tuple(map(int, paper)))
        crop.paste(Image.fromarray(image[top:bottom, l:r]), (0, top - t))
    else:
        crop = Image.fromarray(image[t:b, l:r])
    # A paragraph bounding box can contain letters assigned to another speaker.
    erase = None
    if len(indices) <= MAX_REGIONS:
        for j in indices:
            left, top, right, bottom = boxes[j]
            if j != i and left < r and l < right and top < b and t < bottom:
                if erase is None:
                    erase = Image.new('L', crop.size)
                    draw = ImageDraw.Draw(erase)
                for quad in blocks[j].lines:
                    draw.polygon([(int(x) - l, int(y) - t) for x, y in quad], fill=255)
        if erase is None:
            return crop
    else:  # In a crowded page bound work to this paragraph's own geometry.
        erase = Image.new('L', crop.size, 255)
        draw = ImageDraw.Draw(erase)
    for quad in blocks[i].lines:
        draw.polygon([(int(x) - l, int(y) - t) for x, y in quad], fill=0)
    if paper is None:
        paper = _paper(image[t:b, l:r])
    crop.paste(tuple(map(int, paper)), (0, 0, *crop.size), erase)
    return crop
