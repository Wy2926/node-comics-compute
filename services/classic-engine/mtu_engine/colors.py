"""Sample lettering colors from source RGB and the detector's existing stroke mask."""
import cv2
import numpy as np

from classic_node.timing import record
from .backend import check_cancelled


def _grid(width, height):
    u, v = np.meshgrid((np.arange(width, dtype=np.float32) + .5) / width,
                       (np.arange(height, dtype=np.float32) + .5) / height)
    return np.stack(((1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v), axis=-1)


_PROBE, _DETAIL = _grid(8, 8), _grid(64, 24)
_BLACK, _WHITE = (0, 0, 0), (255, 255, 255)


def _luminance(color):
    # Match the integer RGB channels ultimately passed to Qt.
    rgb = [int(channel) / 255 for channel in color]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return sum(c * weight for c, weight in zip(linear, (0.2126, 0.7152, 0.0722)))


def ensure_stroke_contrast(region):
    """Keep the fill; replace only low-contrast outlines (below 3:1)."""
    foreground, background = _luminance(region.fg_colors), _luminance(region.bg_colors)
    if (max(foreground, background) + 0.05) / (min(foreground, background) + 0.05) < 3:
        black_contrast = (foreground + 0.05) / 0.05
        white_contrast = 1.05 / (foreground + 0.05)
        region.bg_colors = np.full(3, 0 if black_contrast >= white_contrast else 255)
    # Prevent upstream's gray-color heuristic from overriding this decision or
    # dropping the outline. This also covers older checkpoints without a flag.
    region.adjust_bg_color = False


def _points(rgb, quad, grid):
    points = np.rint(grid @ quad).reshape(-1, 2)
    x = points[:, 0].clip(0, rgb.shape[1] - 1).astype(np.intp)
    y = points[:, 1].clip(0, rgb.shape[0] - 1).astype(np.intp)
    return y, x


def _upper(values, fraction):
    k = max(1, len(values) // fraction)
    return np.partition(values, -k)[-k]


def _simple(pixels):
    paper = np.median(pixels, axis=0)
    distance = np.abs(pixels - paper).max(axis=1)
    ink = pixels[distance >= max(32, _upper(distance, 8))]
    if len(ink) < 3:
        return (_WHITE, _BLACK) if _luminance(paper) < 0.18 else (_BLACK, _WHITE)
    return np.median(ink, axis=0).astype(np.uint8), paper.astype(np.uint8)


def _gray(pixels, mask):
    gray = pixels[:, 1]
    outside = gray[mask < 64]
    paper = int(np.median(outside if len(outside) >= 8 else gray))
    low, high = map(int, np.partition(gray, (2, -3))[[2, -3]])
    ink = low if paper - low >= high - paper else high
    if abs(paper - ink) < 32:
        return (_BLACK, _WHITE) if paper >= 118 else (_WHITE, _BLACK)
    return (ink,) * 3, (paper,) * 3


def _masked(rgb, raw, quad, probe):
    y, x = _points(rgb, quad, _DETAIL)
    colors = rgb[y, x].reshape(24, 64, 3)
    mask = raw[y, x].reshape(24, 64)
    outside = colors[mask < 64]
    if len(outside) < 80:
        return _simple(probe)
    paper = np.median(outside, axis=0).astype(np.int16)
    delta = np.abs(colors - paper).max(axis=2)
    strokes = (mask >= 64) & (delta >= 32)
    if np.count_nonzero(strokes) < 48:
        return _simple(probe)
    depth = cv2.distanceTransform(strokes.astype(np.uint8), cv2.DIST_L2, 3)
    high = _upper(depth[strokes], 6)
    core = colors[(depth >= high) & (mask >= 128)]
    if not len(core):
        return _simple(probe)
    fill = np.median(core, axis=0).astype(np.int16)
    shell = colors[strokes & (depth <= max(1.4, high * .6))].astype(np.int16)
    if len(shell):
        # Antialiased pixels mix fill with paper; a distinct outline departs from that RGB segment.
        ray = (fill - paper).astype(np.float32)
        offset = shell - paper
        t = np.clip((offset * ray).sum(axis=1) / max(1., (ray * ray).sum()), 0, 1)
        rim = shell[np.abs(offset - t[:, None] * ray).max(axis=1) > 28]
        if len(rim) >= max(12, len(shell) // 5):
            outline = np.median(rim, axis=0).astype(np.int16)
            core = colors[(depth >= high) & (mask >= 192)]
            if len(core) >= 8:
                distance = np.abs(core - outline).max(axis=1)
                distinct = core[distance >= max(32, _upper(distance, 3))]
                if len(distinct) >= max(4, len(core) // 10):
                    fill = np.median(distinct, axis=0).astype(np.int16)
            return fill, outline
    return fill, paper


def _region_colors(rgb, raw, lines):
    if not len(lines):
        return (_BLACK, _WHITE), False
    best, score = None, -1
    for i in dict.fromkeys((0, len(lines) // 2, len(lines) - 1)):
        quad = np.asarray(lines[i], dtype=np.float32)
        if quad.shape != (4, 2) or not np.isfinite(quad).all():
            continue
        y, x = _points(rgb, quad, _PROBE)
        pixels = rgb[y, x]
        mask = raw[y, x] if raw is not None else None
        r, g, b = pixels.T
        chroma = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b) > 22
        count = np.count_nonzero(chroma)
        strength = np.count_nonzero(chroma & (mask >= 128)) if count and mask is not None else count
        if strength > score:
            best, score = (quad, pixels, mask, count), strength
    if best is None:
        return (_BLACK, _WHITE), False
    quad, pixels, mask, colored = best
    if mask is None:
        return _simple(pixels), False
    if colored < 3:
        return _gray(pixels, mask), False
    if mask.max() < 128:
        return _simple(pixels), False
    ab, bc = quad[1] - quad[0], quad[2] - quad[1]
    if ab @ ab < bc @ bc:
        quad = np.roll(quad, 1, axis=0)  # Align the long side to the 64-sample axis.
    return _masked(rgb, raw, quad, pixels), True


def sample_colors(rgb, regions, raw=None):
    detail = 0
    if raw is not None and raw.shape != rgb.shape[:2]:
        raw = None
    for region in regions:
        check_cancelled()
        pair, refined = _region_colors(rgb, raw, region.lines)
        region.set_font_colors(*pair)
        detail += refined
    record('color_sample_regions', len(regions))
    record('color_detail_regions', detail)
