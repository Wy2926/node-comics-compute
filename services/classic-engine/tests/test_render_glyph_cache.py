"""Real Qt outline reuse: exact geometry, physical fonts and bounded ownership."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

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
