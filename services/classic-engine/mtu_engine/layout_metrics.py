"""Reuse numeric Qt measurements only for the duration of one page."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache


_current = ContextVar('layout_measurements', default=None)


@contextmanager
def page_measurements():
    @lru_cache(maxsize=256)
    def measure(calculate, retain, font, text, args):
        return retain(calculate(text, *args))

    token = _current.set(measure)
    try:
        yield
    finally:
        measure.cache_clear()
        _current.reset(token)


def _measured(calculate, retain, text, args):
    measure = _current.get()
    if measure is None or len(text) > 256:
        return calculate(text, *args)
    from manga_translator.rendering.text_render._fonts import _state
    state = _state()
    font = (state.registry_revision, state.font_family, state.font_style, state.bold)
    return measure(calculate, retain, font, text, args)


def _ink_bounds(values):
    return {k: values[k] for k in ('left_rel', 'top_rel', 'width', 'height', 'logical_width',
                                   'ascent', 'descent', 'has_ink')}


def measured_ink(text, *args):
    from manga_translator.rendering.text_render._layout import _line_ink_geometry
    return _measured(_line_ink_geometry, _ink_bounds, text, args)


def _block(text, size, horizontal, spacing, stroke, letters):
    from manga_translator.rendering import text_render
    return text_render.measure_rich_text_metrics(size, text, horizontal, spacing,
                                                stroke_width=stroke, letter_spacing=letters)


def block_metrics(size, text, horizontal, spacing, *, config, stroke_width, letter_spacing):
    from manga_translator.rendering import text_render
    from manga_translator.rendering.text_render._render import _resolve_stroke_ratio
    if isinstance(text, str) and _current.get() is not None:
        stroke = _resolve_stroke_ratio(config, stroke_width)
        return _measured(_block, dict.copy, text, (size, horizontal, spacing, stroke, letter_spacing))
    return text_render.measure_rich_text_metrics(size, text, horizontal, spacing, config=config,
                                                stroke_width=stroke_width, letter_spacing=letter_spacing)


def _advances(text, size, spacing):
    from manga_translator.rendering.auto_linebreak import _vert_char_advance
    return (_vert_char_advance(size, char, letter_spacing=spacing) for char in text)


def vertical_advances(text, size, spacing):
    return _measured(_advances, tuple, text, (size, spacing))
