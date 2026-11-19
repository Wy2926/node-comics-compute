"""Bounded RGB sampling, checkpoint retention and rendered outline contrast."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from classic_node.protocol import NodeFailure
from classic_node.timing import collect
from mtu_engine.backend import cancellation
from mtu_engine.colors import ensure_stroke_contrast, sample_colors
from mtu_engine.engine import serialize_region


def region(lines):
    block = SimpleNamespace(lines=lines, texts=['original OCR'])
    def set_colors(fg, bg):
        block.fg_colors, block.bg_colors = fg, bg
    block.set_font_colors = set_colors
    return block


@pytest.mark.parametrize('ink,paper', [((200, 12, 35), (245, 240, 230)), ((245, 244, 241), (19, 18, 20))])
def test_sample_reads_line_rgb_without_changing_image_or_ocr(ink, paper):
    rgb = np.full((72, 96, 3), paper, dtype=np.uint8)
    rgb[24:36, 28:60] = ink
    rgb[24:26, 28:30] = (0, 225, 5)  # An isolated speck should not set the fill.
    rgb.flags.writeable = False
    block = region(np.array([[[16, 12], [80, 12], [80, 48], [16, 48]]]))
    with collect() as timings:
        sample_colors(rgb, [block])
    assert tuple(block.fg_colors) == ink and tuple(block.bg_colors) == paper
    assert timings['color_sample_regions'] == 1 and 'ocr_lock_wait' not in timings
    assert block.texts == ['original OCR']


def test_rotated_line_excludes_pixels_outside_the_quad():
    diamond = np.array([[45, 5], [85, 45], [45, 85], [5, 45]])
    y, x = np.ogrid[:91, :91]
    rgb = np.full((91, 91, 3), (0, 255, 0), np.uint8)
    rgb[np.abs(x - 45) + np.abs(y - 45) <= 40] = (245, 245, 230)
    rgb[np.abs(x - 45) + np.abs(y - 45) <= 16] = (170, 33, 52)
    block = region([diamond])
    sample_colors(rgb, [block])
    assert tuple(block.fg_colors) == (170, 33, 52) and tuple(block.bg_colors) == (245, 245, 230)


@pytest.mark.parametrize('gray,fg', [(15, 255), (240, 0)])
def test_uniform_clipped_line_selects_legible_defaults(gray, fg):
    block = region([[[-20, 0], [21, 0], [21, 22], [-20, 22]]])
    sample_colors(np.full((12, 10, 3), gray, np.uint8), [block])
    assert tuple(block.fg_colors) == (fg,) * 3 and tuple(block.bg_colors) == (255 - fg,) * 3


def test_bad_geometry_falls_back_and_cancellation_stops_between_regions():
    blocks = [region([]), region([np.full((4, 2), np.nan)]), region([])]
    def stop():
        if hasattr(blocks[1], 'fg_colors'):
            raise NodeFailure('LEASE_STOPPED')
    with cancellation(stop), pytest.raises(NodeFailure, match='LEASE_STOPPED'):
        sample_colors(np.zeros((2, 2, 3), np.uint8), blocks)
    for block in blocks[:2]:
        assert block.fg_colors == (0, 0, 0) and block.bg_colors == (255, 255, 255)
    assert not hasattr(blocks[-1], 'fg_colors')


def test_color_checkpoint_preserves_prediction_until_rendering():
    block = SimpleNamespace(lines=np.zeros((1, 4, 2)), texts=['source'], font_size=36,
        angle=0, fg_colors=np.array([210, 40, 60]), bg_colors=np.array([120, 120, 120]),
        prob=.9, _direction='h', default_stroke_width=.1, adjust_bg_color=False,
        line_spacing=1., letter_spacing=1.)
    restored = json.loads(json.dumps(serialize_region(block)))
    assert restored['fg_color'] == [210, 40, 60]
    assert restored['bg_color'] == [120, 120, 120]
    assert restored['default_stroke_width'] == .1
    assert restored['adjust_bg_color'] is False


@pytest.mark.parametrize('fg,bg,expected', [
    ([0, 0, 0], [0, 0, 0], [255, 255, 255]),
    ([5, 8, 12], [18, 22, 28], [255, 255, 255]),
    ([0, 0, 0], [0, 0, 100], [255, 255, 255]),
    ([255, 255, 255], [240, 240, 240], [0, 0, 0]),
    ([210, 40, 60], [205, 38, 58], [255, 255, 255]),
    ([0, 0, 255], [0, 0, 250], [255, 255, 255]),
    ([0, 255, 0], [0, 250, 0], [0, 0, 0]),
    ([255, 255, 0], [250, 250, 0], [0, 0, 0]),
    ([120, 120, 120], [120, 120, 120], [0, 0, 0]),
    ([0, 0, 0], [255, 255, 255], [255, 255, 255]),
    ([245, 245, 245], [20, 40, 130], [20, 40, 130]),
    ([220, 30, 50], [255, 255, 255], [255, 255, 255]),
    # Check the 3:1 boundary after the same integer conversion as Qt.
    ([0, 0, 0], [89.99, 89.99, 89.99], [255, 255, 255]),
    ([0, 0, 0], [90, 90, 90], [90, 90, 90]),
])
@pytest.mark.parametrize('adjust', [False, True])
def test_render_color_guard_preserves_fill_and_only_fixes_low_contrast(fg, bg, expected, adjust):
    block = SimpleNamespace(fg_colors=np.array(fg), bg_colors=np.array(bg),
                            adjust_bg_color=adjust, default_stroke_width=.1)
    for _ in range(2):  # Repeated rendering must not flip colors.
        ensure_stroke_contrast(block)
        np.testing.assert_array_equal(block.fg_colors, fg)
        np.testing.assert_array_equal(block.bg_colors, expected)
        assert not block.adjust_bg_color and block.default_stroke_width == .1
