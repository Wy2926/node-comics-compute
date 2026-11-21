"""Sample lettering colors from source RGB and the detector's existing stroke mask."""
import cv2
import numpy as np

from classic_node.timing import record
from .backend import check_cancelled


def _grid(width, height):
    u, v = np.meshgrid((np.arange(width, dtype=np.float32) + .5) / width,
                       (np.arange(height, dtype=np.float32) + .5) / height)
    return np.stack(((1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v), axis=-1)


_PROBE, _DETAIL, _OUTLINE = _grid(8, 8), _grid(64, 24), _grid(96, 32)
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
        end = 0 if black_contrast >= white_contrast else 255
        ink, stroke = [tuple(map(int, c)) for c in (region.fg_colors, region.bg_colors)]
        if (min(max(c) - min(c) for c in (ink, stroke)) > 22
                and max(abs(a - b) for a, b in zip(ink, stroke)) >= 24):
            # Keep a distinct tint on its original side of the fill when that side can reach 3:1.
            if background > foreground and white_contrast >= 3:
                end = 255
            elif background < foreground and black_contrast >= 3:
                end = 0
            def shade(t):
                return [(v * (255 - t) + end * t + 127) // 255 for v in stroke]

            lo, hi = 0, 255
            while lo < hi:
                t = (lo + hi) // 2
                dark, light = sorted((foreground, _luminance(shade(t))))
                if light + .05 >= 3 * (dark + .05):
                    hi = t
                else:
                    lo = t + 1
            region.bg_colors = np.array(shade(lo))
        else:
            region.bg_colors = np.full(3, end)
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


def _enclosed_ink(crop, rim):
    if not rim.size // 12 < cv2.countNonZero(rim) < 3 * rim.size // 4:
        return None
    ring = cv2.copyMakeBorder(rim, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(ring, None, (0, 0), 128)
    ring = ring[1:-1, 1:-1]
    inside = cv2.erode(np.uint8(ring == 0), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))) != 0
    outside = ring == 128
    if inside.sum() < rim.size // 100 or outside.sum() < rim.size // 8:
        return None
    pixels = crop[inside]
    ink = np.median(pixels, axis=0)
    # Counters in ordinary letters enclose the page rather than ink.
    if (np.max(np.median(np.abs(pixels - ink), axis=0)) > 24
            or np.max(np.abs(ink - np.median(crop[outside], axis=0))) < 24):
        return None
    return ink, np.median(crop[ring == 255], axis=0)


def _outline_colors(rgb, raw, quad, fraction, white):
    # Preserve glyph thickness on a fixed grid rather than squeezing a whole long line.
    segment = quad.copy()
    segment[[1, 2]] = quad[[0, 3]] + fraction * (quad[[1, 2]] - quad[[0, 3]])
    y, x = _points(rgb, segment, _OUTLINE)
    crop = rgb[y, x].reshape(32, 96, 3)
    if white:
        result = _enclosed_ink(crop, cv2.inRange(crop, (220,) * 3, _WHITE))
        if result is not None:
            return result
    # CTD ranks colors inside its stroke mask. Use joint RGB bins so channels stay paired.
    support = raw[y, x]
    samples = crop.reshape(-1, 3)
    inside, outside = support >= 128, support < 64
    ink, paper = samples[inside], samples[outside]
    if len(ink) < 32 or len(paper) < 80:
        return None
    r, g, b = (ink.astype(np.int32) >> 4).T
    counts = np.bincount((r << 8) + (g << 4) + b, minlength=4096)
    for _ in range(2):
        index = int(counts.argmax())
        if counts[index] < len(ink) // 12:
            break
        bins = (index >> 8, index >> 4 & 15, index & 15)
        peak = np.array(bins) * 16 + 8
        rim = cv2.inRange(crop, peak - 30, peak + 30)
        matches = rim.ravel()
        # A color covering most of the outside edges must also prefer the detected strokes.
        covers_edges = np.count_nonzero(rim[[0, -1]]) >= rim.shape[1]
        if not covers_edges or np.mean(matches[inside]) > np.mean(matches[outside]):
            result = _enclosed_ink(crop, rim)
            if result is not None:
                fill, stroke = result
                distance = np.abs(paper - fill).max(axis=1)
                separation = max(32, .18 * np.max(np.abs(fill - stroke)))
                if np.mean(distance < 32) < 1 / 3 and np.max(np.abs(np.median(paper, axis=0) - fill)) >= separation:
                    return result
        counts.reshape(16, 16, 16)[tuple(slice(max(0, i - 2), i + 3) for i in bins)] = 0
    return None


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
        pair = _gray(pixels, mask)
        if pair[0][0] >= 220:
            if np.count_nonzero((mask >= 128) & (pixels[:, 1] < 200)) < 3:
                return pair, False
        elif pixels[:, 1].max() <= pair[1][0] + 32 or mask.max() < 128:
            return pair, False
    elif mask.max() < 128:
        return _simple(pixels), False
    ab, bc = quad[1] - quad[0], quad[2] - quad[1]
    long_sq, short_sq = ab @ ab, bc @ bc
    if long_sq < short_sq:
        quad = np.roll(quad, 1, axis=0)  # Align the long side to the 64-sample axis.
        long_sq, short_sq = short_sq, long_sq
    if colored >= 3:
        pair = _masked(rgb, raw, quad, pixels)
    fraction = min(1., 3 * np.sqrt(short_sq / max(1., long_sq)))
    return _outline_colors(rgb, raw, quad, fraction, min(pair[0]) >= 220) or pair, True


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
