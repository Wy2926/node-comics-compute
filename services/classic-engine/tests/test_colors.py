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


@pytest.mark.parametrize('ink,paper', [(0, 255), (255, 0), (59, 225)])
def test_achromatic_text_keeps_polarity_without_entering_detail(ink, paper, monkeypatch):
    rgb = np.full((50, 200, 3), paper, np.uint8)
    mask = np.zeros(rgb.shape[:2], np.uint8)
    rgb[8:42, 20:45] = rgb[8:42, 80:105] = ink
    mask[8:42, 20:45] = mask[8:42, 80:105] = 255
    rgb[12, 28] = (180, 34, 45)  # A speck shouldn't send a gray paragraph down the color path.
    monkeypatch.setattr('mtu_engine.colors._masked', lambda *args: pytest.fail('unnecessary color pass'))
    block = region([[[0, 0], [199, 0], [199, 49], [0, 49]]])
    with collect() as timings:
        sample_colors(rgb, [block], mask)
    assert block.fg_colors == (ink,) * 3 and block.bg_colors == (paper,) * 3
    assert timings['color_detail_regions'] == 0


def test_masked_color_finds_letter_fill_and_outline_on_later_rotated_line():
    from PIL import Image, ImageDraw, ImageFont
    paper, rim, ink = (227, 223, 193), (20, 20, 25), (211, 60, 92)
    image = Image.new('RGB', (400, 150), paper)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=46)
    draw.text((20, 60), 'HELLO', fill=ink, font=font, stroke_fill=rim, stroke_width=3)
    x, y, right, bottom = draw.textbbox((20, 60), 'HELLO', font=font, stroke_width=3)
    rgb = np.array(image)
    mask = np.where(np.any(rgb != paper, axis=-1), 255, 0).astype(np.uint8)
    rgb[:40] = 255
    rgb[10:30, 16:60] = 0
    mask[10:30, 16:60] = 255
    lines = np.array([[[0, 0], [115, 0], [115, 40], [0, 40]],
                      [[x-4, y-4], [right+4, y-4], [right+4, bottom+4], [x-4, bottom+4]]])
    # Rotate the entire page and preserve the native clockwise quad order.
    lines = np.stack((lines[..., 1], rgb.shape[1]-1-lines[..., 0]), axis=-1)
    lines = np.roll(lines, -1, axis=1)
    block = region(lines.copy())
    rgb, mask = np.rot90(rgb), np.rot90(mask)
    rgb.flags.writeable = mask.flags.writeable = False
    with collect() as timings:
        sample_colors(rgb, [block], mask)
    np.testing.assert_allclose(block.fg_colors, ink, atol=8, rtol=0)
    np.testing.assert_allclose(block.bg_colors, rim, atol=12, rtol=0)
    np.testing.assert_array_equal(block.lines, lines)
    assert block.texts == ['original OCR'] and timings['color_detail_regions'] == 1


@pytest.mark.parametrize('shape,value', [((8, 12), 255), ((40, 180), 0), ((40, 180), 255)])
def test_unusable_mask_falls_back_to_observed_pixels(shape, value):
    rgb = np.full((40, 180, 3), (235, 243, 239), np.uint8)
    rgb[5:35, 30:75] = (41, 99, 193)
    block = region([[[0, 0], [179, 0], [179, 39], [0, 39]]])
    sample_colors(rgb, [block], np.full(shape, value, np.uint8))
    assert tuple(block.fg_colors) == (41, 99, 193) and tuple(block.bg_colors) == (235, 243, 239)


@pytest.mark.parametrize('ink', [(108, 98, 96), (212, 72, 108), (255, 255, 255)])
def test_thick_white_outline_is_not_mistaken_for_white_lettering(ink):
    from PIL import Image, ImageDraw, ImageFont
    paper, white = (160, 145, 140), (255,) * 3
    image = Image.new('RGB', (260, 70), paper)
    draw, font = ImageDraw.Draw(image), ImageFont.load_default(size=35)
    text = 'BOBOBO'
    draw.text((10, 9), text, font=font, fill=ink, stroke_fill=white, stroke_width=3)
    l, t, r, b = draw.textbbox((10, 9), text, font=font, stroke_width=3)
    block = region([[[l-3, t-3], [r+3, t-3], [r+3, b+3], [l-3, b+3]]])
    rgb = np.asarray(image)
    mask = np.any(rgb != paper, axis=2).astype(np.uint8) * 255
    sample_colors(rgb, [block], mask)
    np.testing.assert_allclose(block.fg_colors, ink, atol=3)
    np.testing.assert_allclose(block.bg_colors, paper if ink == white else white, atol=16)


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
