"""Real Qt outline reuse: exact geometry, physical fonts and bounded ownership."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import Mock

import numpy as np
import pytest

from mtu_engine.engine import Renderer
from mtu_engine.fonts import FONT_CHOICES
from test_mtu_integration import assets, offline  # noqa: F401


@pytest.mark.parametrize('font_id', FONT_CHOICES)
def test_outline_reuse_matches_qt_shaping_fallback_and_transforms(assets, offline, monkeypatch, font_id):
    Renderer(*assets)
    from manga_translator.rendering.text_render import _fonts, _layout, _glyphs
    font = assets[0].parent / 'fonts' / FONT_CHOICES[font_id][0]
    _fonts.set_font(str(font))
    cached = _layout._glyph_path
    for size, bold, shear, sx, sy in [(24, False, 0, 1, 1), (37, True, .2, 1.2, .8)]:
        for text in ('HELLO office 123 你好 한글 سَلَام', '   ', ''):
            with _fonts._bold_scope(bold):
                monkeypatch.setattr(_layout, '_glyph_path', lambda raw, glyph: raw.pathForGlyph(glyph))
                expected = _layout._horizontal_glyph_path(text, size, False, 1.1,
                    shear=shear, scale_x=sx, scale_y=sy)[3]
                monkeypatch.setattr(_layout, '_glyph_path', cached)
                for _ in range(2):
                    actual = _layout._horizontal_glyph_path(text, size, False, 1.1,
                        shear=shear, scale_x=sx, scale_y=sy)[3]
                    assert actual == expected
                    a, ax, ay = _glyphs._rasterize_path(actual)
                    b, bx, by = _glyphs._rasterize_path(expected)
                    assert (ax, ay) == (bx, by)
                    np.testing.assert_array_equal(a, b)


def test_outline_keys_and_returned_paths_are_independent_qt_values(assets, offline):
    Renderer(*assets)
    from PyQt6.QtGui import QRawFont
    from manga_translator.rendering.text_render import _fonts, _layout
    _fonts._font_registry_changed()
    font = QRawFont(str(assets[0].parent / 'fonts/NotoSans-Regular.ttf'), 24)
    glyph = font.glyphIndexesForString('A')[0]
    expected = font.pathForGlyph(glyph)
    for _ in range(2):  # A caller can mutate either the miss or hit result.
        path = _layout._glyph_path(font, glyph)
        assert path == expected
        path.translate(100, 100)
        assert _layout._glyph_path(font, glyph) == expected
    original = QRawFont(font)
    font.setPixelSize(48)
    assert _layout._glyph_path(font, glyph) == font.pathForGlyph(glyph)
    assert _layout._glyph_path(original, glyph) == expected
    assert len(_fonts._state().glyph_paths) == 2


def test_outline_lru_limits_eviction_oversized_paths_and_registry_invalidation(assets, offline, monkeypatch):
    Renderer(*assets)
    from PyQt6.QtGui import QRawFont
    from manga_translator.rendering.text_render import _fonts, _layout
    _fonts._font_registry_changed()
    state = _fonts._state()
    font = QRawFont(str(assets[0].parent / 'fonts/NotoSans-Regular.ttf'), 24)
    glyphs = font.glyphIndexesForString('ABCDEFGHIJKLMNOPQRSTUVWXYZ ')
    monkeypatch.setattr(_layout, '_GLYPH_PATH_MAX_ENTRIES', 3)
    monkeypatch.setattr(_layout, '_GLYPH_PATH_MAX_ELEMENTS', 100)
    for glyph in glyphs:
        assert _layout._glyph_path(font, glyph) == font.pathForGlyph(glyph)
        assert len(state.glyph_paths) <= 3
        assert state.glyph_path_elements == sum(path.elementCount() for path in state.glyph_paths.values())
        assert state.glyph_path_elements <= 100
    assert (font, glyphs[0]) not in state.glyph_paths
    # A single over-budget outline stays usable but does not enter the LRU.
    monkeypatch.setattr(_layout, '_GLYPH_PATH_MAX_ELEMENTS', 1)
    size = len(state.glyph_paths)
    assert _layout._glyph_path(font, glyphs[0]) == font.pathForGlyph(glyphs[0])
    assert len(state.glyph_paths) == size
    _fonts._font_registry_changed()
    assert _fonts._state() is state
    assert not state.glyph_paths and state.glyph_path_elements == 0


def test_outline_caches_remain_thread_local(assets, offline):
    Renderer(*assets)
    from PyQt6.QtGui import QRawFont
    from manga_translator.rendering.text_render import _fonts, _layout
    barrier = Barrier(2)
    def draw(size):
        font = QRawFont(str(assets[0].parent / 'fonts/NotoSans-Regular.ttf'), size)
        glyph = font.glyphIndexesForString('A')[0]
        expected = font.pathForGlyph(glyph)
        assert _layout._glyph_path(font, glyph) == expected
        state = _fonts._state()
        barrier.wait(timeout=10)
        assert _layout._glyph_path(font, glyph) == expected
        assert list(state.glyph_paths) == [(font, glyph)]
        return state
    with ThreadPoolExecutor(2) as workers:
        left, right = list(workers.map(draw, (24, 48)))
    assert left is not right and left.glyph_paths is not right.glyph_paths


def test_page_ink_reuses_only_matching_qt_metrics(assets, offline, monkeypatch):
    from mtu_engine.layout_metrics import measured_ink, page_measurements
    Renderer(*assets)
    from manga_translator.rendering.text_render import _fonts, _layout
    text = 'Ａ！سَلَام 12＿\u200d'
    ink = _layout._line_ink_geometry
    counted = Mock(wraps=ink)
    monkeypatch.setattr(_layout, '_line_ink_geometry', counted)
    variants = [('NotoSans-Regular.ttf', False, 22, 0.), ('Mada.ttf', False, 22, 0.),
                ('Mada.ttf', True, 22, 0.), ('Mada.ttf', True, 22, .15), ('Mada.ttf', True, 27, .15)]
    with page_measurements():
        for face, bold, size, shear in variants:
            _fonts.set_font(str(assets[0].parent / 'fonts' / face))
            with _fonts._bold_scope(bold):
                args = text, size, .1, True, 1.2, None, shear, 1.3, .8
                expected = ink(*args)
                for _ in range(2):
                    bounds = measured_ink(*args)
                    assert bounds == {k: expected[k] for k in bounds}
                    assert {'path', 'layout'}.isdisjoint(bounds)
        assert counted.call_count == len(variants)
        with _fonts._bold_scope(bold):
            _fonts._font_registry_changed()
            measured_ink(*args)
        assert counted.call_count == len(variants) + 1


def test_page_advances_are_bounded_and_discarded_on_error(assets, offline, monkeypatch):
    from mtu_engine.layout_metrics import page_measurements, vertical_advances
    Renderer(*assets)
    from manga_translator.rendering import auto_linebreak
    advance = Mock(wraps=auto_linebreak._vert_char_advance)
    monkeypatch.setattr(auto_linebreak, '_vert_char_advance', advance)
    def widths(text):
        return tuple(vertical_advances(text, 12, 1))
    text = '＿سَلَام\u200d'
    expected = widths(text)
    advance.reset_mock()
    with pytest.raises(RuntimeError, match='cancelled'), page_measurements():
        for _ in range(2):
            assert widths(text) == expected
            widths('z' * 257)
        assert advance.call_count == len(text) + 2 * 257
        for i in range(256):
            widths(chr(0x400 + i))
        advance.reset_mock()
        widths(text)
        assert advance.call_count == len(text)  # Evicted.
        raise RuntimeError('cancelled')
    widths(text)
    widths(text)
    assert advance.call_count == 3 * len(text)  # No cache survives the render.


def test_block_metrics_cache_preserves_stroke_direction_and_styled_spans(assets, offline, monkeypatch):
    from mtu_engine.engine import configuration
    from mtu_engine.layout_metrics import block_metrics, page_measurements
    Renderer(*assets)
    from manga_translator.rendering import text_render
    from manga_translator.rendering.rich_text import legacy_line_breaks_to_document, TextStyle
    measure = text_render.measure_rich_text_metrics
    counted = Mock(wraps=measure)
    monkeypatch.setattr(text_render, 'measure_rich_text_metrics', counted)
    config, text = configuration(), 'Office[BR]你好'
    opts = dict(config=config, stroke_width=.14, letter_spacing=1.2)
    with page_measurements():
        for disabled, horizontal in ((False, True), (True, True), (False, False)):
            config.render.disable_font_border = disabled
            counted.reset_mock()
            expected = measure(37, text, horizontal, 1.1, **opts)
            for _ in range(2):
                assert block_metrics(37, text, horizontal, 1.1, **opts) == expected
            assert counted.call_count == 1
        styled = legacy_line_breaks_to_document(text, TextStyle.from_dict({'bold': True}))
        for _ in range(2):
            assert block_metrics(37, styled, True, 1.1, **opts) == measure(37, styled, True, 1.1, **opts)
        assert counted.call_count == 3
