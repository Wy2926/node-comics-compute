"""Sample lettering colors from source RGB and keep the rendered outline readable."""
import numpy as np

from classic_node.timing import record
from .backend import check_cancelled


_GRID = (np.arange(16, dtype=np.float32) + 0.5) / 16
_U, _V = np.meshgrid(_GRID, _GRID)
_WEIGHTS = np.stack(((1 - _U) * (1 - _V), _U * (1 - _V), _U * _V, (1 - _U) * _V), axis=-1).reshape(-1, 4)
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


def _sample_line(rgb, lines):
    if not len(lines):
        return _BLACK, _WHITE
    quad = np.asarray(lines[0], dtype=np.float32)
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        return _BLACK, _WHITE
    # Sample inside the rotated line; scratch space is independent of page size.
    points = np.rint(_WEIGHTS @ quad)
    x = points[:, 0].clip(0, rgb.shape[1] - 1).astype(np.intp)
    y = points[:, 1].clip(0, rgb.shape[0] - 1).astype(np.intp)
    pixels = rgb[y, x]
    paper = np.median(pixels, axis=0)
    distance = np.abs(pixels - paper).max(axis=1)
    # The most distinct eighth approximates the ink; ignore faint noise and isolated specks.
    ink = pixels[distance >= max(32, np.partition(distance, -32)[-32])]
    if len(ink) < 3:
        return (_WHITE, _BLACK) if _luminance(paper) < 0.18 else (_BLACK, _WHITE)
    return np.median(ink, axis=0).astype(np.uint8), paper.astype(np.uint8)


def sample_colors(rgb, regions):
    for region in regions:
        check_cancelled()
        region.set_font_colors(*_sample_line(rgb, region.lines))
    record('color_sample_regions', len(regions))
