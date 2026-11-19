"""Page-stage boundaries with a mock paragraph reader; no GPU or model assets."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import sys
from threading import Event, Lock
from types import SimpleNamespace

import numpy as np
import pytest

from mtu_engine.engine import Engine
from mtu_engine.ocr import Reading
from mtu_engine.backend import cancellation
from classic_node.protocol import NodeFailure


@pytest.fixture
def analysis(monkeypatch):
    engine = Engine.__new__(Engine)
    engine.gpu, engine.backend, engine.device, engine.keep_lang = 0, 'cuda', 'cuda:0', None
    engine._detector_lock, engine._bubble_lock, engine._capture_guard = Lock(), Lock(), nullcontext()
    source = np.full((64, 128, 3), 255, dtype=np.uint8)
    quads = [np.array([[4, y], [120, y], [120, y+12], [4, y+12]]) for y in (4, 24, 44)]
    block = SimpleNamespace(lines=quads, font_size=12, angle=0, src_is_vertical=False)
    calls = []
    def text_block(**kwargs):
        return SimpleNamespace(**kwargs, text=' '.join(kwargs['texts']), source_lang='en')
    async def detect(*args, **kwargs):
        return [SimpleNamespace(pts=q) for q in quads], np.ones(source.shape[:2]), None
    async def recognize(*args):
        return [Reading(['one paragraph', '', ''])]
    async def refine(regions, rgb, raw, **kwargs):
        calls.append('refine')
        return raw
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(device=lambda _: nullcontext())))
    monkeypatch.setitem(sys.modules, 'manga_translator.mask_refinement', SimpleNamespace(dispatch=refine))
    monkeypatch.setitem(sys.modules, 'ballontranslator.utils.textblock', SimpleNamespace(group_output=lambda *args: [block]))
    monkeypatch.setitem(sys.modules, 'manga_translator.utils', SimpleNamespace(TextBlock=text_block,
        build_bubble_mask_from_mangalens_result=lambda result, shape: np.zeros(shape),
        is_valuable_text=lambda text: any(char.isalpha() for char in text)))
    engine.config = SimpleNamespace(detector=SimpleNamespace(detection_size=1280, text_threshold=.5,
        box_threshold=.6, unclip_ratio=2.3), render=SimpleNamespace(stroke_width=.1),
        ocr=SimpleNamespace(min_text_length=1, model_bubble_overlap_threshold=.5), mask_dilation_offset=0, kernel_size=3)
    engine.detector = SimpleNamespace(detect=detect)
    engine.ocr = SimpleNamespace(recognize=recognize)
    monkeypatch.setattr('mtu_engine.engine.sample_colors', lambda *a: calls.append('colors'))
    engine.bubbles = SimpleNamespace(detect=lambda *a, **kw: calls.append('bubbles'))
    return engine, source, quads, calls


@pytest.mark.parametrize('shape', [(11745, 800), (800, 11745), (16000, 320), (320, 16000),
                                  (10000, 1536), (1536, 10000), (512, 512)])
def test_analysis_requests_source_short_side_for_native_detection(analysis, shape):
    engine, _, _, _ = analysis
    rgb = np.zeros((*shape, 3), dtype=np.uint8)
    async def detect(image, size, text_threshold, box_threshold, unclip_ratio, **kwargs):
        assert image is rgb and (size, text_threshold, box_threshold, unclip_ratio) == (1280, .5, .6, 2.3)
        assert kwargs == {'det_rearrange_min_effective_short_side': min(shape)}
        return [], None, None
    async def recognize(*args):
        return [Reading([])]
    engine.detector.detect, engine.ocr.recognize = detect, recognize
    assert engine.analyze(rgb) == ([], None, None, None)


@pytest.mark.parametrize('backend,limit,fail_at', [('cuda', 512, None), ('cuda', 1024, None),
    ('mps', 512, None), ('amd', 512, None), ('amd', 768, None), ('amd', 1024, None), ('amd', 512, 192)])
def test_warmup_prepares_graphs_after_native_block_warmup(analysis, backend, limit, fail_at):
    engine, _, _, calls = analysis
    engine.backend = backend
    engine.config.inpainter = SimpleNamespace(inpainting_size=limit)
    shapes = []
    async def warm(*args):
        calls.append('ocr')
    async def inpaint(image, mask, config, size):
        side = image.shape[0]
        assert image.shape == (side, side, 3) and mask.shape == (side, side)
        assert image.dtype == mask.dtype == np.uint8 and np.any(mask) and np.any(mask == 0)
        assert config is engine.config.inpainter and size == limit
        shapes.append(side)
        if side == fail_at:
            raise RuntimeError('fixture inpaint failed')
    engine.ocr.warmup = warm
    engine.inpainter = SimpleNamespace(inpaint=inpaint,
        model=SimpleNamespace(generator=SimpleNamespace(prepare=lambda: calls.append('graphs'))))
    with pytest.raises(RuntimeError, match='fixture inpaint failed') if fail_at else nullcontext():
        engine.warmup()
    expected = [512] + ([s for s in range(64, limit + 1, 64) if s != 512] if backend == 'amd' else [])
    assert shapes == (expected[:expected.index(fail_at)+1] if fail_at else expected)
    assert calls == ['ocr'] + ([] if fail_at else (['bubbles'] if backend == 'mps' else ['graphs', 'bubbles']))
    assert not engine._detector_lock.locked() and not engine._bubble_lock.locked()


@pytest.mark.parametrize('text,keep_lang,minimum', [('', None, 1), (' ', None, 1), ('...', None, 1),
                                                   ('hello', 'zh', 1), ('a', None, 2)])
def test_no_eligible_paragraph_skips_bubbles_and_erasure(analysis, text, keep_lang, minimum):
    engine, source, _, calls = analysis
    async def recognize(*args):
        return [Reading([text, '', ''])]
    engine.ocr.recognize = recognize
    engine.keep_lang, engine.config.ocr.min_text_length = keep_lang, minimum
    assert engine.analyze(source) == ([], None, None, None)
    assert not calls


def test_failure_releases_locks_and_recognition_keeps_paragraph_geometry(analysis):
    engine, source, quads, calls = analysis
    read = engine.ocr.recognize
    async def fail(*args):
        raise RuntimeError('OCR failed')
    engine.ocr.recognize = fail
    with pytest.raises(RuntimeError, match='OCR failed'):
        engine.analyze(source)
    assert not calls and not engine._detector_lock.locked() and not engine._bubble_lock.locked()
    engine.ocr.recognize = read
    regions, mask, raw, _ = engine.analyze(source)
    np.testing.assert_array_equal(regions[0].lines, quads)
    assert regions[0].texts == ['one paragraph', '', '']
    assert mask is raw and calls == ['colors', 'bubbles', 'refine']


def test_capture_guard_covers_other_gpu_stages_but_not_ocr_or_cpu(analysis, monkeypatch):
    engine, source, _, _ = analysis
    guard = engine._capture_guard = Lock()
    detect, read = engine.detector.detect, engine.ocr.recognize
    async def guarded_detect(*args, **kwargs):
        assert guard.locked()
        return await detect(*args, **kwargs)
    async def unlocked_read(*args):
        assert not guard.locked()
        return await read(*args)
    def colors(*args):
        assert not guard.locked()
    def bubbles(*args, **kwargs):
        assert guard.locked()
    def mask(result, shape):
        assert guard.locked()
        return np.zeros(shape)
    from mtu_engine import engine as implementation
    associate = implementation.associate_bubbles
    def associate_cpu(*args):
        assert not guard.locked()
        return associate(*args)
    monkeypatch.setattr(implementation, 'associate_bubbles', associate_cpu)
    monkeypatch.setattr(implementation, 'sample_colors', colors)
    monkeypatch.setattr(sys.modules['manga_translator.utils'], 'build_bubble_mask_from_mangalens_result', mask)
    engine.detector.detect, engine.ocr.recognize, engine.bubbles.detect = guarded_detect, unlocked_read, bubbles
    assert engine.analyze(source)[0] and not guard.locked()


@pytest.mark.parametrize('cancel', [False, True])
def test_inpaint_guards_model_blocks_and_drains_cancellation(analysis, monkeypatch, cancel):
    engine, source, _, _ = analysis
    guard = engine._capture_guard = Lock()
    engine.config.inpainter = SimpleNamespace(inpainting_size=512)
    calls = []
    monkeypatch.setattr(sys.modules['manga_translator.utils'], 'erode_bubble_mask', lambda mask, _: mask, raising=False)
    def fill(rgb, mask, *args, **kwargs):
        assert not guard.locked()
        return rgb, mask, None
    async def blocks(rgb, mask, callback):
        assert not guard.locked()
        for _ in range(3):
            rgb = await callback(rgb, mask)
        return rgb, 3
    async def inpaint(rgb, *args):
        assert guard.locked()
        calls.append(1)
        return rgb.copy()
    monkeypatch.setitem(sys.modules, 'manga_translator.inpainting.ballon_fill', SimpleNamespace(
        MODEL_BUBBLE_SHRINK_RATIO=.02, solid_fill_pure_bubbles=fill, inpaint_regions_per_block=blocks))
    engine.inpainter = SimpleNamespace(inpaint=inpaint)
    mask = np.zeros(source.shape[:2], dtype=np.uint8)
    def check():
        if cancel and calls:
            raise NodeFailure('LEASE_STOPPED')
    with cancellation(check), pytest.raises(NodeFailure, match='LEASE_STOPPED') if cancel else nullcontext():
        np.testing.assert_array_equal(engine.inpaint(source, mask, mask, mask, []), source)
    assert len(calls) == (1 if cancel else 3) and not guard.locked()
    np.testing.assert_array_equal(engine.inpaint(source, mask, mask, mask, []), source)


def test_other_page_can_read_while_sampling_colors(analysis, monkeypatch):
    engine, source, _, _ = analysis
    entered, release = Event(), Event()
    def colors(image, *args, **kwargs):
        if image is source:
            entered.set()
            assert release.wait(5)
    monkeypatch.setattr('mtu_engine.engine.sample_colors', colors)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(engine.analyze, source)
        try:
            assert entered.wait(5)
            assert pool.submit(engine.analyze, source.copy()).result(timeout=5)[0]
        finally:
            release.set()
        assert first.result(timeout=5)[0]
