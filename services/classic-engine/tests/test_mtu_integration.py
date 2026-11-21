"""Real Qt/GPU acceptance; MTU_TEST_ASSETS opts in, MTU_TEST_BACKEND defaults to CUDA."""
from concurrent.futures import ThreadPoolExecutor
import math
import json
import os
from pathlib import Path
import re
import socket
from threading import Barrier

import numpy as np
import pytest

from classic_node.runtime import LANGUAGE_PROBES
from mtu_engine.engine import Engine, Renderer, serialize_region
from mtu_engine.fonts import DEFAULT_FONT_IDS, FONT_CHOICES
from classic_node.config import ENGINE_DEFAULTS


@pytest.fixture(scope='module')
def assets():
    value = os.environ.get('MTU_TEST_ASSETS')
    if not value:
        pytest.skip('Set MTU_TEST_ASSETS to run real upstream components')
    root = Path(value).resolve()
    from mtu_engine.assets import verify
    verify(root / 'models')
    fonts = json.loads((root / 'licenses/font-sources.json').read_text(encoding='utf-8'))['fonts']
    return root / 'models', [str(root / 'fonts' / item['name']) for item in fonts]


def deny_network(*args, **kwargs):
    raise AssertionError('Prepared image stages must work offline')


def test_font_search_prepares_plain_text_once_and_preserves_native_sizes(assets, monkeypatch):
    import inspect
    from mtu_engine.assets import LOCK
    from mtu_engine.engine import configuration
    Renderer(*assets)
    from manga_translator import rendering
    from manga_translator.rendering.text_render import _render
    patch = next(p for p in LOCK['source']['patches']['manga_translator/rendering/__init__.py']
                 if 'def _fits(fs:' in p['before'])
    source = inspect.getsource(rendering.calc_font_from_box)
    assert source.count(patch['after']) == 1
    scope = dict(vars(rendering))
    exec(source.replace(patch['after'], patch['before']), scope)
    native = scope['calc_font_from_box']
    convert, calls = _render.legacy_line_breaks_to_document, []

    def counted(text, *args, **kwargs):
        calls.append(text)
        return convert(text, *args, **kwargs)

    monkeypatch.setattr(_render, 'legacy_line_breaks_to_document', counted)
    monkeypatch.setattr(rendering, 'legacy_line_breaks_to_document', counted)
    for text in ('THIS PERSON[BR]COULD USE MAGIC.', '你好，世界。\n第二行', 'مرحبا بالعالم', 'A'):
        for horizontal in (True, False):
            options = dict(width=230, height=95, text=text, is_horizontal=horizontal,
                           config=configuration(), stroke_width=.1)
            expected = native(**options)
            calls.clear()
            assert rendering.calc_font_from_box(**options) == expected
            assert calls == [text]


def test_style_value_copy_keeps_all_mutable_children_independent(assets):
    import copy
    Renderer(*assets)
    from manga_translator.rendering.rich_text import TextStyle
    style = TextStyle.from_dict({'bold': True, 'italic': 12, 'color': '#123456',
        'stroke': {'color': '#111111', 'width': .1},
        'outerStroke': {'color': '#ffffff', 'width': .2},
        'glow': {'color': '#444444', 'blur': 3}, 'transform': {'rotation': 15, 'scaleX': 2}})
    result = style.copy()
    assert result == copy.deepcopy(style) and result is not style
    for name in ('stroke', 'outer_stroke', 'glow', 'transform'):
        assert getattr(result, name) is not getattr(style, name)
    result.stroke.color = '#aaaaaa'
    result.outer_stroke.width = 1
    result.glow.blur = 9
    result.transform.scale_x = 3
    assert style.stroke.color == '#111111' and style.outer_stroke.width == .2
    assert style.glow.blur == 3 and style.transform.scale_x == 2
    assert TextStyle().copy() == TextStyle()


@pytest.mark.parametrize('shape', [(80, 150), (800, 800), (777, 1103)])
def test_lama_scaling_preserves_thin_masks_and_original_paste_boundary(assets, shape):
    from types import SimpleNamespace
    Renderer(*assets)
    from manga_translator.inpainting.inpainting_lama_mpe import LamaLargeInpainter, resize_keep_aspect
    image = np.full((*shape, 3), 29, np.uint8)
    mask = np.zeros(shape, np.uint8)
    mask[:, shape[1] // 3] = 255
    mask[0, :shape[1] // 2] = 255
    mask[-1, -10:] = 255
    original = mask.copy()
    mask.flags.writeable = False
    owner = SimpleNamespace(logger=SimpleNamespace(info=lambda *args: None))
    prep = LamaLargeInpainter._prepare_large_image(owner, image, mask, 512)
    if max(shape) > 512:
        coverage = resize_keep_aspect(mask.astype(np.float32), 512) > 0
        assert np.all(prep['mask_resized'][:, :, 0][coverage] > 0)
    model_output = np.full(prep['image_pad'].shape, 230, np.uint8)
    result = LamaLargeInpainter._restore_large_output(owner, model_output, prep)
    assert np.all(result[mask > 0] == 230)
    np.testing.assert_array_equal(result[mask == 0], image[mask == 0])
    np.testing.assert_array_equal(mask, original)


def test_inpaint_crop_copy_cleanup_preserves_overlapping_window_order(assets):
    import asyncio
    import inspect
    from mtu_engine.assets import LOCK
    Renderer(*assets)
    from manga_translator.inpainting import ballon_fill
    source = inspect.getsource(ballon_fill.inpaint_regions_per_block)
    for patch in reversed(LOCK['source']['patches']['manga_translator/inpainting/ballon_fill.py']):
        if patch['after'] in source:
            source = source.replace(patch['after'], patch['before'])
    scope = dict(vars(ballon_fill))
    exec(source, scope)
    image = np.random.default_rng(43).integers(0, 256, (70, 90, 3), dtype=np.uint8)
    snapshots, outputs = [], []
    for fn in (scope['inpaint_regions_per_block'], ballon_fill.inpaint_regions_per_block):
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[1:14, 0:12] = 255
        mask[16:29, 14:27] = 255  # Disjoint masks, overlapping enlarged windows.
        mask[-10:, -9:] = 255
        original = image.copy()
        seen = []

        async def repair(crop, local_mask):
            assert not np.shares_memory(crop, image)
            seen.append((crop.copy(), local_mask.copy()))
            crop[:] = (crop.astype(np.uint16) + 13) % 256
            return crop

        result, count = asyncio.run(fn(image, mask, repair))
        assert count == 3 and not np.any(mask)
        np.testing.assert_array_equal(image, original)
        snapshots.append(seen)
        outputs.append(result)
    np.testing.assert_array_equal(*outputs)
    for native, current in zip(*snapshots):
        for a, b in zip(native, current):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('height,width', [(16000,333), (333,16000), (10000,1536), (1536,10000),
                                        (28007,930), (930,28007)])
@pytest.mark.parametrize('long_axis', [False, True])
def test_native_rearrange_preserves_resolution_overlap_and_roundtrip(assets, height, width, long_axis):
    from mtu_engine.assets import activate
    activate(assets[0])
    from manga_translator.utils.generic import (build_det_rearrange_plan,
        det_rearrange_patch_array, det_rearrange_patch_spans, det_unrearrange_patch_maps)
    source = np.zeros((height, width, 3), dtype=np.uint8)
    plan = build_det_rearrange_plan(source, 1280, min(height, width), long_axis=long_axis)
    base_size = max(1280, plan['pw_num'] * min(height, width))
    assert plan['patch_size'] == min(max(height, width), base_size * (2 if long_axis else 1))
    if min(height, width) == 930:
        assert plan['p_num'] == (13 if long_axis else 28)
    packed = det_rearrange_patch_array(plan)
    assert min(packed.shape[1:3]) == plan['pw_num'] * min(height, width)
    spans = det_rearrange_patch_spans(plan)
    assert spans[0][0] == 0 and spans[-1][1] == max(height, width)
    assert all(left[1] - right[0] >= base_size // 5 for left, right in zip(spans, spans[1:]))
    # Feed coordinate ramps through native packing and feathered stitching,
    # including horizontal transposition and the padded tail of the last batch.
    view = np.broadcast_to(np.arange(plan['h'], dtype=np.float32)[:, None, None],
                           (plan['h'], plan['w'], 1))
    coordinate_image = np.swapaxes(view, 0, 1) if plan['transpose'] else view
    coordinate_plan = build_det_rearrange_plan(coordinate_image, 1280, min(height, width), long_axis=long_axis)
    patches = det_rearrange_patch_array(coordinate_plan)
    restored = det_unrearrange_patch_maps(list(patches), coordinate_plan, data_format='hwc')
    np.testing.assert_allclose(restored, coordinate_image, atol=.002, rtol=0)
    assert build_det_rearrange_plan(source[:400,:400],1280) is None


@pytest.mark.parametrize('shape', [(2801, 93), (93, 2801), (1700, 32), (32, 1700),
                                 (1000, 153), (200, 100)])
@pytest.mark.parametrize('batch_size', [1, 4])
def test_streamed_detection_preserves_inputs_and_maps_with_bounded_padding(assets, monkeypatch, shape, batch_size):
    import hashlib
    import inspect
    import weakref
    from mtu_engine.assets import LOCK, activate
    activate(assets[0])
    from manga_translator.utils import generic
    source = inspect.getsource(generic.det_rearrange_forward)
    for patch in reversed(LOCK['source']['patches']['manga_translator/utils/generic.py'][1:]):
        if patch['after'] in source:
            assert source.count(patch['after']) == 1
            source = source.replace(patch['after'], patch['before'])
    scope = dict(vars(generic))
    exec(source, scope)
    image = np.random.default_rng(43).integers(0, 256, (*shape, 3), dtype=np.uint8)
    image.flags.writeable = False
    inputs, outputs, padded = [], [], []
    pad, merge = generic.square_pad_resize, generic.det_unrearrange_patch_maps

    def bounded_pad(*args, **kwargs):
        result = pad(*args, **kwargs)
        padded.append(weakref.ref(result[0]))
        assert sum(ref() is not None for ref in padded) <= batch_size + 1
        return result

    def compact_merge(maps, *args, **kwargs):
        assert all(array.base is None and array.flags.c_contiguous for array in maps)
        return merge(maps, *args, **kwargs)

    original = image.tobytes()
    for fn in (scope['det_rearrange_forward'], generic.det_rearrange_forward):
        seen = []
        def forward(batch, device):
            assert device == 'cpu' and batch.shape[1:] == (128, 128, 3)
            seen.append((batch.shape, hashlib.sha256(batch.tobytes()).hexdigest()))
            # Different output resolutions/channels exercise both crop calculations.
            return (batch[:, ::2, ::2, :2].transpose(0, 3, 1, 2).astype(np.float32),
                    batch[:, ::4, ::4, 2:3].transpose(0, 3, 1, 2).astype(np.float32))
        outputs.append(fn(image, forward, 128, batch_size, device='cpu',
                          min_effective_short_side=min(shape)))
        inputs.append(seen)
        assert image.tobytes() == original
        monkeypatch.setattr(generic, 'square_pad_resize', bounded_pad)
        monkeypatch.setattr(generic, 'det_unrearrange_patch_maps', compact_merge)
    assert inputs[0] == inputs[1]
    for old, new in zip(*outputs, strict=True):
        np.testing.assert_array_equal(old, new)
    assert all(ref() is None for ref in padded)
    plan = generic.build_det_rearrange_plan(image, 128, min(shape))
    if plan is None:
        assert inputs == [[], []] and outputs == [(None, None), (None, None)]
    else:
        assert len(padded) == plan['p_num']
        assert [shape[0] for shape, _ in inputs[1]] == [
            min(batch_size, plan['p_num'] - i) for i in range(0, plan['p_num'], batch_size)]
        if min(shape) == 32:
            assert plan['pw_num'] > 1 and plan['pad_num'] > 0


@pytest.mark.parametrize('failure_type', ['cancel', 'model'])
@pytest.mark.parametrize('stride', [0, 32])
def test_streamed_detection_releases_batches_after_failure_and_can_resume(assets, monkeypatch, failure_type, stride):
    import gc
    import weakref
    from classic_node.protocol import NodeFailure
    from mtu_engine.assets import activate
    from mtu_engine.backend import cancellation, check_cancelled
    activate(assets[0])
    from manga_translator.utils import generic
    refs, calls = [], []
    pad = generic.square_pad_resize
    def tracked_pad(*args, **kwargs):
        result = pad(*args, **kwargs)
        refs.append(weakref.ref(result[0]))
        return result
    monkeypatch.setattr(generic, 'square_pad_resize', tracked_pad)
    def forward(batch, **kwargs):
        check_cancelled()
        if calls and failure_type == 'model':
            raise RuntimeError('model failed')
        calls.append(len(batch))
        maps = batch[:, ::2, ::2, :1].transpose(0, 3, 1, 2).astype(np.float32)
        refs.extend((weakref.ref(batch), weakref.ref(maps)))
        return maps, maps
    def check():
        if calls and failure_type == 'cancel':
            raise NodeFailure('LEASE_STOPPED')
    image = np.zeros((2801, 93, 3), np.uint8)
    def run():
        return generic.det_rearrange_forward(image, forward, 128, 2, min_effective_short_side=93,
                                             input_stride=stride)
    with pytest.raises(NodeFailure if failure_type == 'cancel' else RuntimeError) as failure:
        with cancellation(check):
            run()
    assert calls == [1 if stride else 2] and len(refs) == (2 if stride else 6)
    del failure  # Release pytest's retained traceback before checking buffer ownership.
    gc.collect()
    assert all(ref() is None for ref in refs)
    failure_type = None
    db, mask = run()
    assert db.shape == mask.shape == (1, 1, 1400, 47)
    assert not db.any() and not mask.any()
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize('shape', [(2801, 93), (93, 2801), (1700, 32), (32, 1700),
                                  (1000, 153), (153, 1000)])
@pytest.mark.parametrize('batch_size', [1, 4])
def test_rectangular_detection_preserves_scale_and_bounds_batch_pixels(assets, shape, batch_size):
    import cv2
    from mtu_engine.assets import activate
    activate(assets[0])
    from manga_translator.utils.generic import (det_rearrange_forward, build_det_rearrange_plan,
        det_rearrange_patch_array, det_unrearrange_patch_maps)
    image = np.random.default_rng(13).integers(0, 256, (*shape, 3), dtype=np.uint8)
    plan = build_det_rearrange_plan(image, 128, min(shape), long_axis=True)
    expected, db_maps, mask_maps, seen = [], [], [], []
    for patch in det_rearrange_patch_array(plan):
        scale = min(1, 128 / min(patch.shape[:2]))
        if scale < 1:
            patch = cv2.resize(patch, (round(patch.shape[1] * scale), round(patch.shape[0] * scale)))
        h, w = patch.shape[:2]
        expected.append(np.pad(patch, ((0, -h % 32), (0, -w % 32), (0, 0))))
        db_maps.append(patch[::2, ::2, :2].transpose(2, 0, 1).astype(np.float32))
        mask_maps.append(patch[::4, ::4, 2:3].transpose(2, 0, 1).astype(np.float32))
    def forward(batch, **kwargs):
        n, h, w, _ = batch.shape
        assert h % 32 == w % 32 == 0
        assert n <= min(batch_size, max(1, batch_size * 128 ** 2 // (h * w)))
        if min(shape) == 93:
            assert sorted((h, w)) == [96, 256]
        seen.extend(batch)
        return (batch[:, ::2, ::2, :2].transpose(0, 3, 1, 2).astype(np.float32),
                batch[:, ::4, ::4, 2:3].transpose(0, 3, 1, 2).astype(np.float32))
    outputs = det_rearrange_forward(image, forward, 128, batch_size, input_stride=32,
                                    min_effective_short_side=min(shape))
    np.testing.assert_array_equal(seen, expected)
    for output, maps in zip(outputs, (db_maps, mask_maps), strict=True):
        np.testing.assert_array_equal(output[0], det_unrearrange_patch_maps(maps, plan))


@pytest.fixture
def offline(monkeypatch):
    connect = socket.socket.connect
    def loopback_only(sock, address):
        # Windows asyncio implements its wakeup pipe with a loopback socketpair.
        if isinstance(address, tuple) and address[0] in ('127.0.0.1', '::1'):
            return connect(sock, address)
        return deny_network()
    monkeypatch.setattr(socket.socket, 'connect', loopback_only)
    monkeypatch.setattr(socket, 'create_connection', deny_network)


@pytest.mark.parametrize('horizontal', [False, True])
@pytest.mark.parametrize('shape,budget,count', [((400,400),1000,100), ((1280,1280),1000,1000),
                                              ((4800,400),2000,1200), ((6000,1000),3000,3000)])
def test_detection_candidate_budget_and_model_alignment(assets, monkeypatch, horizontal, shape, budget, count):
    import asyncio
    from mtu_engine.assets import activate
    from mtu_engine.engine import configuration
    activate(assets[0])
    from manga_translator.detection import default
    from manga_translator.detection.default_utils.dbnet_utils import SegDetectorRepresenter
    pred = np.zeros(shape, np.float32)
    for y in range(0, shape[0], 40):
        for x in range(0, shape[1], 40):
            pred[y+8:y+24, x+8:x+24] = .9
    if horizontal:
        pred = pred.T.copy()
    db = pred[None, None]
    shape = pred.shape
    def forward(image, infer, size, batch, **kwargs):
        assert size == 1280 and batch == 2
        assert kwargs['device'] == 'cpu' and kwargs['input_stride'] == 256
        return db, db
    def representer(text_threshold, box_threshold, **kwargs):
        assert (text_threshold, box_threshold, kwargs) == (.5, .6, {'max_candidates': budget, 'unclip_ratio': 2.3})
        return SegDetectorRepresenter(text_threshold, box_threshold, **kwargs)
    monkeypatch.setattr(default, 'det_rearrange_forward', forward)
    monkeypatch.setattr(default.dbnet_utils, 'SegDetectorRepresenter', representer)
    detector = default.DefaultDetector()
    detector.device, detector.max_batch_size = 'cpu', 2
    c = configuration().detector
    lines, mask, _ = asyncio.run(detector._infer(np.zeros((*shape, 3), np.uint8), c.detection_size,
                                               c.text_threshold, c.box_threshold, c.unclip_ratio))
    assert len(lines) == count and mask.shape == shape
    assert all(line.prob > .89 and line.area > 0 for line in lines)


def test_detection_compact_bitmap_preserves_contours_boxes_and_scores(assets, monkeypatch):
    import cv2
    import inspect
    import textwrap
    from types import MethodType
    from mtu_engine.assets import LOCK, activate
    activate(assets[0])
    from manga_translator.detection.default_utils import dbnet_utils
    cls = dbnet_utils.SegDetectorRepresenter
    source = inspect.getsource(cls.boxes_from_bitmap)
    for patch in reversed(LOCK['source']['patches']['manga_translator/detection/default_utils/dbnet_utils.py']):
        assert source.count(patch['after']) == 1
        source = source.replace(patch['after'], patch['before'])
    scope = dict(vars(dbnet_utils))
    exec(textwrap.dedent(source), scope)
    decoder = cls(.5, .6)
    native = MethodType(scope['boxes_from_bitmap'], decoder)
    pred = np.zeros((180, 240), np.float32)
    pred[10:35, 10:80] = .9
    pred[15:25, 20:60] = .1  # Preserve native RETR_LIST handling of holes.
    pred[70:95, 90:150] = .55  # Geometry passes, confidence does not.
    cv2.fillPoly(pred, [np.array([[160, 50], [220, 80], [210, 100], [150, 70]])], .8)
    pred[140, ::3] = .9  # Small noise must stay rejected.
    expected = native(pred, pred > .5, 480, 360)
    find = cv2.findContours
    def contours(bitmap, *args):
        assert bitmap.dtype == np.uint8 and bitmap.max() == 1
        return find(bitmap, *args)
    monkeypatch.setattr(cv2, 'findContours', contours)
    for actual, old in zip(decoder.boxes_from_bitmap(pred, pred > .5, 480, 360), expected):
        np.testing.assert_array_equal(actual, old)


@pytest.mark.parametrize('parts', [0, 2])
def test_detection_skips_empty_or_split_unclip_results(assets, monkeypatch, parts):
    from mtu_engine.assets import activate
    activate(assets[0])
    from manga_translator.detection.default_utils.dbnet_utils import SegDetectorRepresenter
    decoder = SegDetectorRepresenter(.5, .6)
    monkeypatch.setattr(decoder, 'unclip', lambda *a, **kw: np.zeros((parts, 4, 2)))
    pred = np.zeros((30, 50), np.float32)
    pred[5:20, 5:40] = .9
    boxes, scores = decoder.boxes_from_bitmap(pred, pred > .5, 50, 30)
    assert not boxes.any() and not scores.any()


def test_detection_accepts_borderline_box_without_rounding_reported_score(assets):
    from mtu_engine.assets import activate
    activate(assets[0])
    from manga_translator.detection.default_utils.dbnet_utils import SegDetectorRepresenter
    pred = np.zeros((64, 100), np.float32)
    for i, value in enumerate((.5994, .59995, .60044)):
        pred[12:42, 8+i*32:22+i*32] = value
    _, scores = SegDetectorRepresenter(.5, .6).boxes_from_bitmap(pred, pred > .5, 100, 64)
    np.testing.assert_allclose(np.sort(scores[scores > 0]), [.59995, .60044], atol=1e-6)


def test_detection_border_removal_rejects_padding_without_mutating_lines(assets):
    from types import SimpleNamespace
    from mtu_engine.assets import activate
    activate(assets[0])
    from manga_translator.detection.common import CommonDetector
    rows = [(10, 10, 40, 30), (90, 60, 120, 90), (110, 10, 130, 30),
            (10, 90, 40, 110), (-10, 10, 0, 30)]
    lines = [SimpleNamespace(pts=np.array([[x,y], [r,y], [r,b], [x,b]]),
        xyxy=(x,y,r,b), text='fixture', prob=.9, det_label='text') for x,y,r,b in rows]
    snapshots = [line.pts.copy() for line in lines]
    detector = SimpleNamespace(_resize_and_crop_mask=lambda *args: None)
    result, _, _ = CommonDetector._remove_border(
        detector, np.zeros((400, 400, 3), np.uint8), 100, 80, lines, None, None)
    assert len(result) == 2
    assert all(line.det_label == 'text' and line.area > 0 for line in result)
    assert result[1].xyxy == (90, 60, 100, 80)
    for line, original in zip(lines, snapshots):
        np.testing.assert_array_equal(line.pts, original)


@pytest.mark.parametrize('shape,rearrange', [((28007,930), True), ((930,28007), True),
                                            ((4000,320), True), ((300,200), False)])
def test_detection_only_plans_before_border_for_small_inputs(assets, shape, rearrange):
    import asyncio
    from mtu_engine.assets import activate
    activate(assets[0])
    from manga_translator.detection.common import CommonDetector
    checks, sizes = [], []
    class Detector(CommonDetector):
        def _should_rearrange_before_border(self, *args):
            checks.append(1)
            return rearrange
        async def _detect(self, image, *args):
            sizes.append(image.shape[:2])
            return [], None, None
    asyncio.run(Detector().detect(np.zeros((*shape, 3), np.uint8), 1280, .5, .6, 2.3))
    assert len(checks) == int(min(shape) < 400)
    assert sizes == [shape if rearrange else (400, 400)]


def test_staggered_vertical_captions_keep_reading_order_and_the_short_aside(assets):
    Renderer(*assets)
    from ballontranslator.utils.textblock import group_output

    def column(x, top, bottom):
        return np.array([[x, top], [x+20, top], [x+20, bottom], [x, bottom]])

    # Two pairs staggered by five glyphs; a short aside sits inside the last column's vertical span.
    columns = [column(160, 10, 260), column(132, 12, 240), column(104, 110, 370),
               column(76, 112, 325), column(48, 243, 285)]
    def groups(lines):
        return [[line[0][0] for line in block.lines] for block in group_output([], lines, 220, 400)]

    assert groups(columns[::-1]) == [[160, 132], [104, 76], [48]]
    for top, bottom in [(10, 210), (85, 185), (190, 260)]:  # top-, center-, and bottom-aligned dialogue
        assert groups([column(132, top, bottom), columns[0]]) == [[160, 132]]


@pytest.mark.parametrize('horizontal', [False, True])
@pytest.mark.parametrize('font_size', [18, 26])
def test_long_page_cross_seam_lines_are_complete_without_duplicate_ocr(assets, offline, monkeypatch,
                                                                      horizontal, font_size):
    import asyncio
    import torch
    from PIL import Image, ImageDraw, ImageFont
    models, fonts = assets
    Renderer(models, fonts)
    from manga_translator.detection import default
    from manga_translator.utils import ModelWrapper
    from manga_translator.utils.generic import build_det_rearrange_plan, det_rearrange_patch_spans
    from ballontranslator.utils.textblock import group_output
    from mtu_engine.engine import configuration
    from mtu_engine.ocr.hayai import HayaiOCR
    backend = os.environ.get('MTU_TEST_BACKEND', 'cuda')
    device = 'mps' if backend == 'mps' else 'cuda'
    ModelWrapper._MODEL_DIR = str(models)
    detector = default.DefaultDetector()
    asyncio.run(detector.load(device))
    native = default.det_rearrange_forward
    # This focused regression can run beside an existing service. It retains
    # native tile geometry, inference and merging, but limits GPU batch size.
    monkeypatch.setattr(default, 'det_rearrange_forward',
        lambda image, forward, size, batch, **kw: native(image, forward, size, 1, **kw))
    english = None
    try:
        assert next(detector.model.parameters()).dtype == (torch.float16 if device == 'mps' else torch.float32)
        # This length places cuts exactly 256px apart from adjacent tile ends.
        image = Image.new('RGB', (14080, 720) if horizontal else (720, 14080), 'white')
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(str(models.parent / 'fonts/NotoSans-Regular.ttf'), font_size)
        sentence = 'TALKING ABOUT MAGIC'
        bounds = draw.textbbox((0, 0), sentence, font=font)
        seams = [top for top, _ in det_rearrange_patch_spans(
            build_det_rearrange_plan(np.asarray(image), 1280, min_effective_short_side=720,
                                     long_axis=True))[1:]]
        assert seams[0] == 2560 - 256
        # Center a labeled line on an actual cut, independent of tile geometry.
        first_y = seams[1] - 20 * 40 - (bounds[1] + bounds[3]) // 2
        count = 15 if horizontal else 30
        for index in range(count):
            position = ((seams[1] - (bounds[0] + bounds[2]) // 2, 40 + index * 40)
                        if horizontal else (180, first_y + index * 40))
            draw.text(position, sentence, font=font, fill='black')
        rgb = np.array(image)
        config = configuration()
        c = config.detector
        assert (c.text_threshold, c.box_threshold, c.unclip_ratio) == (.5, .6, 2.3)
        lines, raw, _ = asyncio.run(detector.detect(rgb, c.detection_size, c.text_threshold,
            c.box_threshold, c.unclip_ratio, det_rearrange_min_effective_short_side=720))
        getattr(torch, device).empty_cache()
        blocks = group_output([], [line.pts for line in lines], image.width, image.height)
        assert len(lines) == count and sum(len(block.lines) for block in blocks) == count
        english = HayaiOCR(models / 'hayai', backend=backend)
        # One crop per detected line isolates seams from paragraph grouping.
        from types import SimpleNamespace
        crops = [SimpleNamespace(xyxy=[*line.pts.min(axis=0), *line.pts.max(axis=0)], lines=[line.pts]) for line in lines]
        readings = asyncio.run(english.recognize(rgb, crops, config.ocr))
        assert [row.texts[0].upper() for row in readings] == [sentence] * count
        axis = 0 if horizontal else 1
        assert any(block.xyxy[axis] < seam < block.xyxy[axis + 2] for block in blocks for seam in seams)
        assert raw.shape == rgb.shape[:2]
        from manga_translator.mask_refinement import dispatch
        from manga_translator.utils import TextBlock
        regions = [TextBlock(lines=block.lines, texts=[sentence]) for block in blocks]
        mask = asyncio.run(dispatch(regions, rgb, raw, dilation_offset=config.mask_dilation_offset,
                                    kernel_size=config.kernel_size))
        assert np.all(mask[rgb[:, :, 0] < 245] == 255)  # Includes small antialiased seam glyphs.
    finally:
        if english is not None:
            english.close()
        asyncio.run(detector.unload())
        default.MODEL = None
        getattr(torch, device).empty_cache()


def test_detector_fusion_preserves_fp32_predictions_and_rejects_unknown_normalization(assets, offline, monkeypatch):
    import asyncio
    import torch
    from mtu_engine.assets import activate
    from mtu_engine.detection import fuse_detector, prepare_detector
    activate(assets[0])
    from manga_translator.detection import default
    from manga_translator.utils import ModelWrapper
    ModelWrapper._MODEL_DIR = str(assets[0])
    detector = default.DefaultDetector()
    monkeypatch.setattr(default, 'prepare_detector', lambda model, *_: model)
    asyncio.run(detector.load('cpu'))
    try:
        assert sum(isinstance(m, torch.nn.BatchNorm2d) for m in detector.model.modules()) == 70
        batch = np.random.default_rng(71).integers(0, 256, (1, 256, 256, 3), dtype=np.uint8)
        raw, mask = default.det_batch_forward_default(batch, 'cpu')
        assert raw.shape[1] == 2
        expected = raw[:, :1], mask
        prepare_detector(detector.model, 'cpu', 'fp32')
        for left, right in zip(default.det_batch_forward_default(batch, 'cpu'), expected, strict=True):
            np.testing.assert_array_equal(left, right)
        assert fuse_detector(detector.model) is detector.model
        actual = default.det_batch_forward_default(batch, 'cpu')
        assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in detector.model.modules())
        for left, right in zip(actual, expected, strict=True):
            assert left.dtype == right.dtype == np.float32
            np.testing.assert_allclose(left, right, atol=1e-3, rtol=1e-3)
        with pytest.raises(ValueError, match='Unsupported detector normalization'):
            fuse_detector(torch.nn.ModuleDict({'unsupported': torch.nn.BatchNorm2d(3)}).eval())
    finally:
        asyncio.run(detector.unload())
        default.MODEL = None


def test_detector_fusion_runs_in_half_on_selected_gpu(assets, offline, monkeypatch):
    import asyncio
    import torch
    from mtu_engine.assets import activate
    from mtu_engine import detection
    activate(assets[0])
    from manga_translator.detection import default
    from manga_translator.utils import ModelWrapper
    device = 'mps' if os.environ.get('MTU_TEST_BACKEND') == 'mps' else 'cuda'
    ModelWrapper._MODEL_DIR = str(assets[0])
    detector = default.DefaultDetector()
    fold = detection.fuse_conv_bn_eval
    folded_dtypes = set()
    def fold_fp32(conv, bn, **kwargs):
        folded_dtypes.update((conv.weight.dtype, bn.running_var.dtype))
        return fold(conv, bn, **kwargs)
    monkeypatch.setattr(detection, 'fuse_conv_bn_eval', fold_fp32)
    asyncio.run(detector.load(device, precision='fp16'))
    try:
        weight = next(detector.model.parameters())
        assert folded_dtypes == {torch.float32}
        assert weight.device.type == device and weight.dtype == torch.float16
        if device == 'cuda' and torch.version.cuda:
            assert weight.is_contiguous(memory_format=torch.channels_last)
        assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in detector.model.modules())
        batch = np.random.default_rng(3).integers(0, 256, (1, 256, 256, 3), dtype=np.uint8)
        for prediction in default.det_batch_forward_default(batch, device):
            assert prediction.shape[1] == 1 and prediction.dtype == np.float32
            assert np.isfinite(prediction).all()
    finally:
        asyncio.run(detector.unload())
        default.MODEL = None


@pytest.mark.parametrize('precision', ['float16', 'float32'])
def test_detector_inputs_follow_model_and_predictions_remain_fp32(assets, monkeypatch, precision):
    import torch
    from mtu_engine.assets import activate
    activate(assets[0])
    from manga_translator.detection import default
    dtype = getattr(torch, precision)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1, dtype=dtype))

        def forward(self, batch):
            assert batch.dtype == dtype and not torch.is_grad_enabled()
            return batch[:, :2], batch[:, :1]

    monkeypatch.setattr(default, 'MODEL', Model())
    batch = np.arange(2 * 32 * 64 * 3, dtype=np.uint8).reshape(2, 32, 64, 3)[:, :, ::-1]
    db, mask = default.det_batch_forward_default(batch, 'cpu')
    expected = torch.from_numpy((batch.astype(np.float32) / 127.5 - 1).transpose(0, 3, 1, 2)).to(dtype).float()
    np.testing.assert_allclose(db, expected[:, :2].sigmoid().numpy(), atol=1e-7, rtol=0)
    np.testing.assert_array_equal(mask, expected[:, :1].numpy())
    assert db.dtype == mask.dtype == np.float32


def test_qt_worker_multilingual_text_and_region_fit_offline(bubble_layout):
    renderer, calls, _ = bubble_layout
    from manga_translator.utils import TextBlock
    block = TextBlock(np.array([[[90, 70], [510, 70], [510, 300], [90, 300]]]),
                      ['source'], font_size=48, default_stroke_width=0.07)
    region = serialize_region(block)
    restored = TextBlock(**region)
    assert restored.stroke_width == block.stroke_width
    source = np.full((400, 600, 3), 255, dtype=np.uint8)
    mask = np.zeros(source.shape[:2], dtype=np.uint8)
    sentence = 'THIS PERSON COULD WIELD MAGIC TO CONTROL ALL THINGS.'
    word_samples = {'en': sentence, 'ar': 'كان هذا الشخص قادرًا على تسخير السحر للتحكم في كل شيء.',
                    'fr': 'CETTE PERSONNE POUVAIT TOUT CONTRÔLER PAR LA MAGIE.'}
    with ThreadPoolExecutor(2) as workers:
        outputs = []
        for language, probe in LANGUAGE_PROBES.items():
            assert renderer.covers(probe), language
            text = word_samples.get(language, probe)
            output = workers.submit(renderer.render, source, source, [region], [text], language, mask).result()
            changed = np.any(output != source, axis=2)
            assert changed.any(), language
            y, x = np.where(changed)
            # smart_scaling permits a small expansion beyond the source box.
            assert x.min() >= 80 and x.max() <= 520 and y.min() >= 60 and y.max() <= 310, language
            outputs.append(output)
        english = [item for item in calls if item.target_lang == 'ENG'][0]
        assert re.sub(r'\s*\[BR\]\s*', ' ', english.translation).split() == sentence.split()
        from mtu_engine.engine import LANGUAGES
        for language, sentence_text in word_samples.items():
            block = next(item for item in calls if item.target_lang == LANGUAGES[language])
            assert re.sub(r'\s*\[BR\]\s*', ' ', block.translation).split() == sentence_text.split(), language
        repeated = workers.submit(renderer.render, source, source, [region], [sentence], 'en', mask).result()
        assert np.array_equal(repeated, outputs[list(LANGUAGE_PROBES).index('en')])
        empty = workers.submit(renderer.render, source, source, [region], [''], 'en', mask).result()
        assert np.array_equal(empty, source)


@pytest.mark.parametrize('font_id', FONT_CHOICES)
def test_selected_regular_or_light_font_uses_its_physical_face_and_fallback(assets, offline, font_id):
    from PyQt6.QtGui import QRawFont
    renderer = Renderer(*assets)
    from manga_translator.rendering.text_render import _fonts
    filename, languages = FONT_CHOICES[font_id]
    path = renderer.font_by_name[filename]
    sample = 'مرحبًا بالعالم' if languages[0] == 'ar' else LANGUAGE_PROBES[languages[0]]
    expected = QRawFont(path, 48)
    assert expected.isValid() and expected.weight() <= 400
    _fonts.set_font(path)
    _, _, layout, _ = _fonts._create_text_layout(sample, 48)
    runs = layout.glyphRuns()
    assert runs and all(index != 0 for run in runs for index in run.glyphIndexes())
    assert all(run.rawFont().familyName() == expected.familyName() for run in runs)
    assert all(run.rawFont().styleName() == expected.styleName() and run.rawFont().weight() <= 400 for run in runs)
    assert all(run.rawFont().weight() == (300 if font_id.endswith('-light') else 400) for run in runs)
    # A mixed-script name must stay visible even when absent from the chosen face.
    fallback_sample = sample + (' مُحَمَّد' if expected.supportsCharacter(ord('龘')) else ' 龘')
    _, _, mixed, _ = _fonts._create_text_layout(fallback_sample, 48)
    assert all(index != 0 for run in mixed.glyphRuns() for index in run.glyphIndexes())
    assert any(run.rawFont().familyName() != expected.familyName() for run in mixed.glyphRuns())


def test_selected_fonts_cover_all_languages_and_do_not_mix_between_render_threads(assets, offline):
    renderer = Renderer(*assets)
    from manga_translator.utils import TextBlock
    block = TextBlock(np.array([[[70, 60], [530, 60], [530, 260], [70, 260]]]),
                      ['source'], font_size=36, direction='h')
    region = serialize_region(block)
    source = np.full((320, 600, 3), 255, np.uint8)
    choices = [(font_id, language) for font_id, (_, languages) in FONT_CHOICES.items() for language in languages]
    assert set(renderer.font_ids) == set(FONT_CHOICES)

    def draw(choice):
        font_id, language = choice
        return renderer.render(source, source, [region], [LANGUAGE_PROBES[language]], language, None, font_id=font_id)

    expected = [draw(choice) for choice in choices]
    assert all(np.any(result != source) for result in expected)
    with ThreadPoolExecutor(3) as workers:
        actual = list(workers.map(draw, choices * 2))
    for result, reference in zip(actual, expected * 2):
        assert np.array_equal(result, reference)
    assert not np.array_equal(draw(('comic-relief', 'en')), draw(('patrick-hand', 'en')))


def test_unselected_null_and_empty_fonts_render_each_language_default(assets, offline, monkeypatch):
    renderer = Renderer(*assets)
    from manga_translator.utils import TextBlock
    block = TextBlock(np.array([[[70, 60], [530, 60], [530, 260], [70, 260]]]),
                      ['source'], font_size=36, direction='h')
    region = serialize_region(block)
    source = np.full((320, 600, 3), 255, np.uint8)
    selected = []
    render_blocks = renderer._render_blocks

    def capture(original, cleaned, blocks, language, config, *args):
        selected.append((language, config.render.font_family, [block.font_family for block in blocks]))
        return render_blocks(original, cleaned, blocks, language, config, *args)

    monkeypatch.setattr(renderer, '_render_blocks', capture)
    for language, default in DEFAULT_FONT_IDS.items():
        args = (source, source, [region], [LANGUAGE_PROBES[language]], language, None)
        expected = renderer.render(*args, font_id=default)
        assert np.any(expected != source), language
        assert np.array_equal(renderer.render(*args), expected), language
        for font_id in (None, ''):
            assert np.array_equal(renderer.render(*args, font_id=font_id), expected), language
        filename = renderer.font_by_name[FONT_CHOICES[default][0]]
        assert selected[-4:] == [(language, filename, [filename])] * 4


def test_renderer_rejects_unknown_or_uninstalled_default_without_first_font_fallback(assets, offline):
    renderer = Renderer(*assets)
    source = np.full((30, 40, 3), 255, np.uint8)
    args = (source, source, [], [], 'en', None)
    with pytest.raises(ValueError, match='Unsupported translation font'):
        renderer.render(*args, font_id='unknown')
    renderer.font_by_name.pop(FONT_CHOICES[DEFAULT_FONT_IDS['en']][0])
    for font_id in (None, '', DEFAULT_FONT_IDS['en']):
        with pytest.raises(ValueError, match='Translation font is not installed'):
            renderer.render(*args, font_id=font_id)


def test_font_choices_survive_real_render_process_pool(assets, offline):
    from classic_node.render_pool import RenderPool, render_page
    renderer = Renderer(*assets)
    from manga_translator.utils import TextBlock
    source = np.full((240, 420, 3), 255, np.uint8)
    block = TextBlock(np.array([[[40, 40], [380, 40], [380, 200], [40, 200]]]),
                      ['source'], font_size=30, direction='h')
    regions = [serialize_region(block)]
    options = {'alpha': None, 'version': 'font-fixture', 'analysis': {'input_hash': '0' * 64},
               'translated': {'analysis_hash': '1' * 64, 'revision': '2' * 64}}
    choices = [('en', 'comic-relief'), ('en', 'patrick-hand'), ('en', None), ('en', ''),
               ('zh-Hans', 'lxgw-wenkai-light'), ('zh-Hans', 'lxgw-wenkai'),
               ('zh-Hans', 'noto-sans-cjk-light'), ('zh-Hans', 'noto-sans-cjk'),
               ('ko', 'gaegu'), ('ar', 'noto-naskh-arabic'), ('ar', ''), ('ja', None)]
    expected = [render_page(renderer, source, source, regions, [LANGUAGE_PROBES[language]], language, None,
                            font_id=font_id, **options) for language, font_id in choices]
    pool = RenderPool(*assets, workers=2, threads=1)
    try:
        pool.warmup()

        def draw(choice):
            language, font_id = choice
            return pool.render(source, source, regions, [LANGUAGE_PROBES[language]], language, None,
                               font_id=font_id, **options)

        with ThreadPoolExecutor(3) as workers:
            assert list(workers.map(draw, choices * 2)) == expected * 2
    finally:
        pool.close()


def private_layout_cases():
    manifest = os.environ.get('MTU_TEST_LAYOUT_CASES')
    if not manifest:
        return [pytest.param(None, id='private-pages-disabled')]
    path = Path(manifest).resolve()
    return [pytest.param((path.parent, row), id=Path(row['image']).stem)
            for row in json.loads(path.read_text(encoding='utf-8'))]


@pytest.mark.parametrize('turns', range(4))
@pytest.mark.parametrize('foreground', [(15, 15, 15), (220, 35, 40)])
def test_ctd_refinement_keeps_outlines_edges_and_excludes_nearby_text(assets, turns, foreground):
    import asyncio
    import cv2
    from mtu_engine.assets import activate
    activate(assets[0])
    from manga_translator.mask_refinement import dispatch
    from manga_translator.utils import TextBlock
    rgb = np.full((120, 220, 3), (150, 170, 190), np.uint8)
    glyph = np.zeros(rgb.shape[:2], np.uint8)
    outline = np.zeros_like(glyph)
    cv2.putText(glyph, 'TEST', (0, 43), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 255, 2)
    cv2.putText(outline, 'TEST', (0, 43), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 255, 6)
    rgb[outline > 0] = (255, 255, 255)
    rgb[glyph > 0] = foreground
    raw = glyph.copy()
    raw[65:69, 30:80] = 255  # Inside the CTD context window, outside selected text.
    rgb[65:69, 30:80] = foreground
    points = np.array([[0, 13], [100, 13], [100, 49], [0, 49]])
    for _ in range(turns):
        points = np.column_stack((points[:, 1], rgb.shape[1] - 1 - points[:, 0]))
        rgb, raw, outline = (np.rot90(a) for a in (rgb, raw, outline))
    snapshots = rgb.copy(), raw.copy(), points.copy()
    rgb.flags.writeable = raw.flags.writeable = False
    block = TextBlock(lines=[points], texts=['fixture'])
    result = asyncio.run(dispatch([block], rgb, raw, dilation_offset=20))
    assert result.dtype == np.uint8 and result.shape == raw.shape
    assert np.all(result[outline > 0] == 255)
    assert not np.any(np.rot90(result, -turns)[65:69, 30:80])
    for actual, expected in zip((rgb, raw, points), snapshots):
        np.testing.assert_array_equal(actual, expected)


def test_ctd_batched_component_selection_and_fill_match_native(assets):
    import inspect
    from mtu_engine.assets import LOCK, activate
    activate(assets[0])
    from manga_translator.detection.ctd_utils import textmask
    source = inspect.getsource(textmask.merge_mask_list)
    for patch in LOCK['source']['patches']['manga_translator/detection/ctd_utils/textmask.py'][1:]:
        assert source.count(patch['after']) == 1
        source = source.replace(patch['after'], patch['before'])
    scope = dict(vars(textmask))
    exec(source, scope)
    native = scope['merge_mask_list']
    rng = np.random.default_rng(34)
    for probability in (0, .15, .55, 1):
        candidates = [[(rng.random((51, 79)) < probability).astype(np.uint8) * 255, k] for k in (2, 0, 1)]
        pred = rng.integers(0, 256, candidates[0][0].shape, dtype=np.uint8)
        for threshold, mode in ((0, textmask.REFINEMASK_ANNOTATION), (30, textmask.REFINEMASK_INPAINT)):
            args = dict(pred_thresh=threshold, refine_mode=mode)
            np.testing.assert_array_equal(textmask.merge_mask_list(candidates.copy(), pred, **args),
                                          native(candidates.copy(), pred, **args))


def test_ctd_refinement_cancels_between_lines_and_resumes(assets, monkeypatch):
    import sys
    from mtu_engine.assets import activate
    from mtu_engine.backend import cancellation
    from classic_node.protocol import NodeFailure
    from mtu_engine.masks import complete_mask
    assert not any(name.startswith('pydensecrf') for name in sys.modules)
    monkeypatch.setitem(sys.modules, 'pydensecrf', None)
    activate(assets[0])
    from manga_translator.detection.ctd_utils import textmask
    from manga_translator.mask_refinement import complete_mask as dispatched
    from manga_translator.utils import Quadrilateral
    assert dispatched is complete_mask
    rgb = np.full((90, 140, 3), 255, np.uint8)
    mask = np.zeros(rgb.shape[:2], np.uint8)
    mask[20:30, 25:55] = mask[60:70, 25:55] = 255
    rgb[mask > 0] = 0
    lines = [Quadrilateral(np.array([[15,y], [65,y], [65,y+25], [15,y+25]]), '', 0)
             for y in (10, 50)]
    calls, native = [], textmask.refine_mask
    def refine(*args, **kwargs):
        calls.append(True)
        return native(*args, **kwargs)
    monkeypatch.setattr(textmask, 'refine_mask', refine)
    def cancelled():
        if calls:
            raise NodeFailure('LEASE_STOPPED')
    with cancellation(cancelled), pytest.raises(NodeFailure, match='LEASE_STOPPED'):
        complete_mask(rgb, mask, lines)
    assert len(calls) == 1
    resumed = complete_mask(rgb, mask, lines)
    assert np.all(resumed[mask > 0] == 255) and len(calls) == 3


def test_ctd_dispatch_empty_inputs_and_unsupported_method(assets):
    import asyncio
    from mtu_engine.assets import activate
    activate(assets[0])
    from manga_translator.mask_refinement import dispatch
    from manga_translator.utils import TextBlock
    rgb = np.full((40, 60, 3), 255, np.uint8)
    raw = np.zeros(rgb.shape[:2], np.uint8)
    block = TextBlock(lines=[[[5, 5], [40, 5], [40, 30], [5, 30]]], texts=['fixture'])
    assert not asyncio.run(dispatch([block], rgb, raw)).any()
    raw[10:20, 10:30] = 255
    assert not asyncio.run(dispatch([], rgb, raw)).any()
    with pytest.raises(ValueError, match='Unsupported mask refinement method'):
        asyncio.run(dispatch([block], rgb, raw, method='fill'))


def test_ctd_refinement_keeps_complete_glyph_cut_by_detection_polygon(assets):
    import cv2
    from mtu_engine.assets import activate
    from mtu_engine.masks import complete_mask
    activate(assets[0])
    from manga_translator.utils import Quadrilateral
    rgb = np.full((90, 140, 3), 170, np.uint8)
    glyph = np.zeros(rgb.shape[:2], np.uint8)
    glyph[20:65, 20:30] = glyph[20:65, 55:65] = glyph[40:46, 20:65] = 255
    outline = cv2.dilate(glyph, np.ones((5, 5), np.uint8))
    rgb[outline > 0], rgb[glyph > 0] = 255, 0
    raw = glyph.copy()
    raw[30:40, 83:89] = 255
    rgb[30:40, 83:89] = 0  # Nearby disconnected text is not selected.
    line = Quadrilateral(np.array([[20,16], [38,16], [38,70], [20,70]]), '', 0)
    result = complete_mask(rgb, raw, [line], dilation_offset=20)
    assert np.all(result[outline > 0] == 255)
    assert not np.any(result[30:40, 83:89])


@pytest.mark.parametrize('case', private_layout_cases())
def test_private_page_refinement_preserves_detected_text(assets, case):
    if case is None:
        pytest.skip('Set MTU_TEST_LAYOUT_CASES to a private page manifest')
    import asyncio
    import cv2
    from PIL import Image
    from classic_node.runtime import decode_mask
    from mtu_engine.engine import configuration
    Renderer(*assets)
    from manga_translator.mask_refinement import dispatch, _build_line_protect_mask
    from manga_translator.utils import TextBlock
    root, row = case
    original = np.asarray(Image.open(root / row['image']).convert('RGB'))
    analysis = json.loads((root / row['analysis']).read_text(encoding='utf-8'))
    raw, bubbles = [decode_mask(analysis, key, original.shape[1::-1]) for key in ('raw_mask', 'bubble_mask')]
    blocks = [TextBlock(**r) for r in analysis['regions']]
    config = configuration()
    mask = asyncio.run(dispatch(blocks, original, raw, bubble_mask=bubbles,
        dilation_offset=config.mask_dilation_offset, kernel_size=config.kernel_size,
        limit_mask_dilation_to_bubble_mask=True))
    support = cv2.dilate(cv2.compare(raw, 0, cv2.CMP_GT), None, iterations=2)
    selected = _build_line_protect_mask(blocks, original.shape[:2], expand_px=2 + config.kernel_size // 2)
    assert np.all(mask[(support > 0) & (selected > 0)] == 255)
    assert not np.shares_memory(mask, raw) and not np.shares_memory(mask, bubbles)
    # A fresh production checkpoint must use exactly the tested repair mask.
    np.testing.assert_array_equal(mask, decode_mask(analysis, 'mask', original.shape[1::-1]))


@pytest.mark.parametrize('case', private_layout_cases())
@pytest.mark.parametrize('language', ['zh-Hans', 'en'])
@pytest.mark.parametrize('cache_bytes', [0, None])
def test_private_page_layout_replays_fixed_checkpoints(bubble_layout, case, language, cache_bytes):
    """Opt-in private pages: real Qt, unchanged GPU checkpoint and fixed text."""
    if case is None:
        pytest.skip('Set MTU_TEST_LAYOUT_CASES to a private page manifest')
    import hashlib
    import cv2
    from PIL import Image
    from classic_node.runtime import decode_mask
    from manga_translator.utils import build_region_reference_mask, erode_bubble_mask
    root, row = case
    if row.get('language', language) != language:
        pytest.skip('Private fixture supplies another target language')
    source_path = root / row['image']
    analysis = json.loads((root / row['analysis']).read_text(encoding='utf-8'))
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == analysis['input_hash']
    original = np.asarray(Image.open(source_path).convert('RGB'))
    cleaned = np.asarray(Image.open(root / row['cleaned']).convert('RGB'))
    repair_mask = decode_mask(analysis, 'mask', original.shape[1::-1])
    np.testing.assert_array_equal(cleaned[repair_mask == 0], original[repair_mask == 0])
    # Optional clean-only ROIs catch old glyphs before the new text can hide them.
    # Keep private pixels/coordinates outside the repository; textured ROIs can
    # specify their own background tolerance instead of requiring blanket white.
    for check in row.get('clean_checks', []):
        x1, y1, x2, y2 = check['bbox']
        dark = np.max(cleaned[y1:y2, x1:x2], axis=2) < check.get('dark_below', 160)
        assert np.count_nonzero(dark) <= check['max_dark_pixels'], check['bbox']
    translated = json.loads((root / row['translations']).read_text(encoding='utf-8'))[source_path.name]
    texts = [translated[s['id']] if language == row.get('language', 'zh-Hans') else s['source']
             for s in analysis['segments']]
    inset = erode_bubble_mask(decode_mask(analysis, 'bubble_mask', original.shape[1::-1]), .02)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(inset, connectivity=8)
    renderer, observed, native = bubble_layout
    result = renderer.render(original, cleaned, analysis['regions'], texts, language, inset,
        bubble_prepared=True, bubble_components=(labels, stats), mask_cache_bytes=cache_bytes)
    assert result.shape == original.shape and np.any(result != cleaned)
    assert len(observed) == len(texts)
    expected = {tuple(np.asarray(region['lines']).ravel()): text
                for region, text in zip(analysis['regions'], texts, strict=True)}
    allowed = inset.copy()
    height, width = original.shape[:2]
    for index, block in enumerate(observed):
        text = expected[tuple(block.lines.ravel())]  # Source geometry must not be rewritten.
        actual = re.sub(r'\s*\[BR\]\s*', ' ' if language == 'en' else '', block.translation)
        # Normal words stay intact. A narrow-caption recovery may introduce
        # line-end hyphens; joining those must restore the original words.
        joined = re.sub(r'-\[BR\]', '', block.translation).replace('[BR]', ' ')
        expected_tokens = re.findall(r'\w+|[^\w\s]', text)
        assert (re.findall(r'\w+|[^\w\s]', actual) == expected_tokens or
                (language == 'en' and re.findall(r'\w+|[^\w\s]', joined) == expected_tokens))
        assert block.font_size > 1
        points = block.dst_points.reshape(-1, 2)
        assert (points >= (1, 1)).all() and (points <= (width - 1, height - 1)).all()
        if block.bubble_enclosed:
            reference = build_region_reference_mask(block, inset, labels)
            assert native._polygon_fully_inside_mask(points, reference)
        else:
            cv2.fillPoly(allowed, [np.rint(points).astype(np.int32)], 255)
        for neighbor in observed[index+1:]:
            assert not native._polygons_overlap(points, neighbor.dst_points)
    assert not np.any(np.any(result != cleaned, axis=2) & (allowed == 0))


@pytest.mark.parametrize(('language', 'text'), [
    ('en', 'THIS PERSON COULD WIELD MAGIC TO CONTROL ALL THINGS.'),
    ('zh-Hans', '这个人能够运用魔法，控制世间的一切事物。'),
    ('zh-Hant', '這個人能夠運用魔法，控制世間的一切事物。'),
    ('ja', 'この人は魔法を使って、すべてのものを操ることができた。'),
    ('ko', '이 사람은 마법을 사용하여 모든 것을 제어할 수 있었습니다.'),
    ('ar', 'كان هذا الشخص قادرًا على تسخير السحر للتحكم في كل شيء.'),
    ('fr', 'CETTE PERSONNE POUVAIT TOUT CONTRÔLER PAR LA MAGIE.'),
    ('es', 'ESTA PERSONA PODÍA CONTROLAR TODO CON MAGIA.'),
    ('pt-BR', 'ESTA PESSOA PODIA CONTROLAR TODAS AS COISAS COM MAGIA.'),
    ('de', 'DIESE PERSON KONNTE MIT MAGIE ALLES KONTROLLIEREN.'),
    ('it', 'QUESTA PERSONA POTEVA CONTROLLARE TUTTO CON LA MAGIA.'),
    ('ru', 'Этот человек мог управлять всем с помощью магии.'),
    ('pl', 'Ta osoba mogła kontrolować wszystko za pomocą magii.'),
    ('uk', 'Ця людина могла керувати всім за допомогою магії.'),
    ('tr', 'Bu kişi sihir kullanarak her şeyi kontrol edebilirdi.'),
    ('vi', 'Người này có thể dùng phép thuật để điều khiển mọi thứ.'),
    ('id', 'Orang ini bisa mengendalikan segalanya dengan sihir.'),
])
def test_qt_mask_optimization_preserves_pixels_and_layout(bubble_layout, monkeypatch, language, text):
    import cv2
    from mtu_engine.assets import LOCK
    renderer, observed, rendering = bubble_layout
    from manga_translator.utils import TextBlock
    from manga_translator.utils.bubble import _reference_masks
    source = np.full((1800, 720, 3), 240, dtype=np.uint8)
    mask = np.zeros(source.shape[:2], dtype=np.uint8)
    regions = []
    for top in (90, 650, 1200):
        mask[top:top + 450, 80:650] = 255
        block = TextBlock([[[260, top + 70], [430, top + 70], [430, top + 370], [260, top + 370]]],
                          ['source'], font_size=48, direction='v')
        regions.append(serialize_region(block))
    def snapshot():
        return [(block.translation, block.font_size, block.dst_points.tolist()) for block in observed]
    optimized = rendering._polygon_fully_inside_mask
    optimized_rectangle = rendering.find_largest_inscribed_rect
    namespace = {'np': np, 'cv2': cv2}
    patches = LOCK['source']['patches']['manga_translator/rendering/__init__.py']
    exec(patches[0]['before'], namespace)
    rectangle = next(p for p in patches if p['before'].startswith('def find_largest_inscribed_rect('))
    exec(rectangle['before'], namespace)
    monkeypatch.setattr(rendering, '_polygon_fully_inside_mask', namespace['_polygon_fully_inside_mask'])
    monkeypatch.setattr(rendering, 'find_largest_inscribed_rect', namespace['find_largest_inscribed_rect'])
    baseline = renderer.render(source, source, regions, [text] * 3, language, mask, mask_cache_bytes=0)
    layout = snapshot()
    observed.clear()
    monkeypatch.setattr(rendering, '_polygon_fully_inside_mask', optimized)
    monkeypatch.setattr(rendering, 'find_largest_inscribed_rect', optimized_rectangle)
    candidate = renderer.render(source, source, regions, [text] * 3, language, mask)
    assert np.array_equal(baseline, candidate)
    assert snapshot() == layout
    assert _reference_masks.get() is None


# Exact luminance thresholds are covered without Qt in test_colors.py.
@pytest.mark.parametrize('enclosed,matching_background,adjust,fg,bg,stroke', [
    (False, True, None, [0, 0, 0], [10, 10, 10], [255, 255, 255]),
    (True, True, True, [255, 255, 255], [240, 240, 240], [0, 0, 0]),
    (True, False, False, [245, 245, 245], [20, 40, 130], [20, 40, 130]),
    (False, False, True, [210, 40, 60], [205, 38, 58], [255, 255, 255]),
])
def test_qt_guards_stroke_contrast_after_checkpoint_restore(
        assets, offline, enclosed, matching_background, adjust, fg, bg, stroke):
    models, fonts = assets
    renderer = Renderer(models, fonts)
    from manga_translator.utils import TextBlock
    block = TextBlock([[[100, 80], [500, 80], [500, 210], [100, 210]]], ['source'],
        font_size=48, fg_color=fg, bg_color=bg, default_stroke_width=.1, adjust_bg_color=False)
    data = json.loads(json.dumps(serialize_region(block)))
    if adjust is None:
        data.pop('adjust_bg_color')
    else:
        data['adjust_bg_color'] = adjust
    # Also match the page to the fill: only the outline keeps text visible.
    cleaned = np.full((300, 600, 3), fg if matching_background else [240, 220, 190], dtype=np.uint8)
    bubble = np.zeros(cleaned.shape[:2], dtype=np.uint8)
    if enclosed:
        bubble[30:260, 40:560] = 255
    result = renderer.render(cleaned, cleaned, [data], ['COLOR TEST'], 'en', bubble)
    assert np.count_nonzero(np.all(result == fg, axis=2)) > 20
    assert np.count_nonzero(np.all(result == stroke, axis=2)) > 20
    assert data['fg_color'] == fg and data['bg_color'] == bg
    assert data.get('adjust_bg_color') is adjust  # Do not mutate saved checkpoints.


def test_legacy_checkpoint_retains_black_white_without_color_inference(assets, offline):
    models, fonts = assets
    renderer = Renderer(models, fonts)
    from manga_translator.utils import TextBlock
    block = TextBlock([[[100, 80], [500, 80], [500, 210], [100, 210]]], ['source'], font_size=48)
    data = serialize_region(block)
    data.pop('adjust_bg_color')
    cleaned = np.full((300, 600, 3), 220, dtype=np.uint8)
    result = renderer.render(cleaned, cleaned, [data], ['LEGACY TEST'], 'en', None)
    assert np.count_nonzero(np.all(result == [0, 0, 0], axis=2)) > 20
    assert np.count_nonzero(np.all(result == [255, 255, 255], axis=2)) > 20


@pytest.mark.parametrize('fill,background', [
    ((200, 20, 40), (245, 240, 230)), ((255, 255, 255), (0, 0, 0)), ((0, 0, 0), (255, 255, 255))])
def test_sampled_lettering_colors_survive_native_checkpoint(assets, fill, background):
    from PIL import Image, ImageDraw, ImageFont
    from mtu_engine.assets import activate
    from mtu_engine.colors import sample_colors
    models, _ = assets
    activate(models)
    from manga_translator.utils import TextBlock
    image = Image.new('RGB', (500, 140), background)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(models.parent / 'fonts/NotoSans-Regular.ttf'), 48)
    draw.text((35, 30), 'PLAIN TEXT', font=font, fill=fill)
    x1, y1, x2, y2 = draw.textbbox((35, 30), 'PLAIN TEXT', font=font)
    quad = [[x1-4, y1-4], [x2+4, y1-4], [x2+4, y2+4], [x1-4, y2+4]]
    block = TextBlock([quad, quad], ['PLAIN TEXT', ''], language='en', adjust_bg_color=False)
    rgb = np.asarray(image)
    mask = np.any(rgb != background, axis=2).astype(np.uint8) * 255
    sample_colors(rgb, [block], mask)
    restored = TextBlock(**serialize_region(block))
    assert restored.get_font_colors() == (fill, background)
    assert restored.texts == block.texts and np.array_equal(restored.lines, block.lines)


def test_cuda_models_warmup_and_local_inpaint_offline(assets, offline, monkeypatch):
    models, fonts = assets
    backend = os.environ.get('MTU_TEST_BACKEND', 'cuda')
    engine = Engine(models, fonts, backend=backend)
    try:
        engine.warmup()
        assert engine.ocr.identity['backend'] == 'hayai'
        assert engine.ocr.engine.device.type == ('mps' if backend == 'mps' else 'cuda')
        import asyncio
        from types import SimpleNamespace
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype(str(models.parent / 'fonts/NotoSansCJKsc-Regular.otf'), 36)
        for text in ('WHERE ARE YOU GOING?', '明日はきっと晴れる。', '你好世界', '오늘은 날씨가 좋습니다'):
            image = Image.new('RGB', (600, 100), 'white')
            draw = ImageDraw.Draw(image)
            draw.text((12, 8), text, font=font, fill='black')
            x1, y1, x2, y2 = draw.textbbox((12, 8), text, font=font)
            quad = np.array([[x1-3,y1-3],[x2+3,y1-3],[x2+3,y2+3],[x1-3,y2+3]])
            block = SimpleNamespace(xyxy=[x1-3,y1-3,x2+3,y2+3], lines=[quad], src_is_vertical=False)
            row, = asyncio.run(engine.ocr.recognize(np.array(image), [block], engine.config.ocr))
            assert ''.join(row.texts[0].split()).upper() == ''.join(text.split()).upper()
        # Exercise the real detector at the page boundary with bounded paragraph batches.
        shapes = []
        infer = engine.ocr._recognize
        def capture(crops):
            shapes.append(len(crops))
            return infer(crops)
        monkeypatch.setattr(engine.ocr, '_recognize', capture)
        edge_font = ImageFont.truetype(str(models.parent / 'fonts/NotoSans-Regular.ttf'), 26)
        sentence = 'BUT UNLIKE ME'
        edge_inputs = []
        edge_regions = {}
        for count in (17, 4):
            image = Image.new('RGB', (720, 1203), 'white')
            draw = ImageDraw.Draw(image)
            bounds = draw.textbbox((0, 0), sentence, font=edge_font)
            bottom_y = image.height - 1 - bounds[3]
            for index in range(count):
                draw.text((70 if index % 2 else 380, bottom_y - (count - 1 - index) * 64),
                          sentence, font=edge_font, fill='black')
            regions, mask, _, _ = engine.analyze(np.array(image))
            assert len(regions) == count and all(r.text.startswith('BUT ') for r in regions)
            assert mask is not None and mask[-8:].any()
            edge_inputs.append((np.array(image), count))
            edge_regions[count] = [serialize_region(region) for region in regions]
        assert sum(shapes) == sum(edge_regions) and all(1 <= n <= ENGINE_DEFAULTS['ocr_batch_size'] for n in shapes)
        # Pair analyses at the same start gate and compare full checkpoint
        # regions, including line order and colors, against serial inference.
        gate = Barrier(2)
        def analyze_pair(image):
            gate.wait(timeout=30)
            return engine.analyze(image)
        with ThreadPoolExecutor(2) as workers:
            paired = [(count, workers.submit(analyze_pair, image)) for image, count in edge_inputs]
            for count, job in paired:
                regions, *_ = job.result()
                assert [serialize_region(region) for region in regions] == edge_regions[count]
        source = np.full((256, 256, 3), 220, dtype=np.uint8)
        source[110:135, 110:135] = 0
        mask = np.zeros(source.shape[:2], dtype=np.uint8)
        mask[100:145, 100:145] = 255
        cleaned = engine.inpaint(source, mask, mask, np.zeros_like(mask), [])
        assert cleaned.shape == source.shape and cleaned.dtype == np.uint8
        assert np.array_equal(cleaned[mask == 0], source[mask == 0])
        assert np.any(cleaned[mask != 0] != source[mask != 0])
        # Mix different pages and real inpainting on the node's two compute
        # workers. This catches shared upstream state crossing page boundaries.
        with ThreadPoolExecutor(2) as workers:
            jobs = []
            for image, count in edge_inputs * 2:
                jobs.append((count, workers.submit(engine.analyze, image)))
                jobs.append((None, workers.submit(engine.inpaint, source, mask, mask, np.zeros_like(mask), [])))
            for count, job in jobs:
                output = job.result()
                if count is None:
                    assert np.array_equal(output[mask == 0], source[mask == 0])
                    assert np.abs(output.astype(np.int16) - cleaned.astype(np.int16)).max() <= 1
                else:
                    regions, edge_mask, _, _ = output
                    assert [serialize_region(region) for region in regions] == edge_regions[count]
                    assert edge_mask is not None and edge_mask[-8:].any()
        assert all(1 <= n <= ENGINE_DEFAULTS['ocr_batch_size'] for n in shapes)
    finally:
        engine.close()


@pytest.mark.skipif(os.environ.get('MTU_TEST_BACKEND') not in ('cuda', 'amd'), reason='Requires CUDA/HIP validation')
def test_lama_graphs_match_native_pixels_through_cpu_transfer_and_concurrent_pages(assets, offline, monkeypatch):
    import asyncio
    from classic_node.timing import collect
    from mtu_engine.backend import device_context
    from mtu_engine.engine import configuration
    from mtu_engine.inpainting import LamaGraphs
    Renderer(*assets)
    from manga_translator.utils import ModelWrapper
    from manga_translator.inpainting.inpainting_lama_mpe import LamaLargeInpainter
    monkeypatch.setattr(ModelWrapper, '_MODEL_DIR', str(assets[0]))
    class TorchLama(LamaLargeInpainter):
        _MODEL_MAPPING = {'model': LamaLargeInpainter._MODEL_MAPPING['model']}
    native = TorchLama()
    asyncio.run(native.load('cuda', force_torch=True))
    config = configuration().inpainter
    samples = []
    reference_shapes = []
    rng = np.random.default_rng(317)
    def infer(image, mask):
        with device_context(os.environ['MTU_TEST_BACKEND'], 0):
            return asyncio.run(native.inpaint(image, mask, config, 512))
    def reference_transfer(model, pixels, mask, device):
        import torch
        reference_shapes.append(pixels.shape[:2])
        image = torch.from_numpy(pixels).permute(2, 0, 1)[None].float().div(255).to(device)
        mask = torch.from_numpy(mask[:, :, 0])[None, None].float().to(device)
        with torch.no_grad():
            output = model(image.mul_(1 - mask), mask)
        return (output.cpu()[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    try:
        with monkeypatch.context() as reference:
            reference.setattr('mtu_engine.inpainting.cuda_lama_pixels', reference_transfer)
            for height, width in ((64, 64), (128, 128), (257, 257), (512, 512), (777, 777), (80, 150)):
                image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
                mask = np.zeros((height, width), np.uint8)
                mask[height//3:height*2//3, width//4:width*3//4] = 255
                mask[0, :width//3] = 255
                samples.append((image, mask, infer(image, mask)))
        assert len(reference_shapes) == len(samples)  # Both CUDA and HIP must actually select the byte-transfer hook.
        native.model.generator = LamaGraphs(native.model.generator)
        native.model.generator.prepare()
        assert len(native.model.generator.graphs) == 8
        def check(sample):
            image, mask, expected = sample
            with collect() as timings:
                actual = infer(image, mask)
            # CUDA eager itself can round a repaired channel one uint8 level differently.
            tolerance = 1 if os.environ['MTU_TEST_BACKEND'] == 'cuda' else 0
            np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)
            np.testing.assert_array_equal(actual[mask == 0], image[mask == 0])
            key = 'inpaint_graph_blocks' if image.shape[0] == image.shape[1] else 'inpaint_eager_blocks'
            assert timings[key] == 1
        with ThreadPoolExecutor(2) as workers:
            list(workers.map(check, samples * 2))
    finally:
        asyncio.run(native.unload())


def test_missing_cuda_fails_before_loading_an_alternate_backend(monkeypatch):
    import torch
    monkeypatch.setattr(torch.version, 'hip', None)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CPU fallback is disabled'):
        Engine('missing', [])


def test_long_lines_and_partial_ocr_continue_at_different_positions(assets, offline, monkeypatch):
    from PIL import Image, ImageDraw, ImageFont
    from mtu_engine.ocr import Reading
    models, fonts = assets
    engine = Engine(models, fonts, backend=os.environ.get('MTU_TEST_BACKEND', 'cuda'))
    sentences = ['We read a new story together at the library.',
                 'The evening sky was full of distant stars.',
                 'Our next adventure begins early tomorrow.']
    font = ImageFont.truetype(str(models.parent / 'fonts/NotoSans-Regular.ttf'), 24)
    recognize = engine.ocr.recognize
    failure = None
    async def incomplete(*args, **kwargs):
        result = await recognize(*args, **kwargs)
        return [Reading([''] * len(row.texts))
                if failure == 'all-empty' or (failure and 'evening' in ' '.join(row.texts)) else row
                for row in result]
    monkeypatch.setattr(engine.ocr, 'recognize', incomplete)
    try:
        engine.warmup()
        for top in (32, 440, 900):
            image = Image.new('RGB', (800, 1200), 'white')
            draw = ImageDraw.Draw(image)
            for index, sentence in enumerate(sentences):
                draw.text((70, top + index * 90), sentence, font=font, fill='black')
            draw.text((70, 1100 if top < 900 else 200), 'STILL READABLE', font=font, fill='black')
            rgb = np.array(image)
            failure = None
            regions, mask, _, _ = engine.analyze(rgb)
            recognized = ' '.join(region.text for region in regions)
            for anchor in ('We read a new', 'evening sky', 'Our next adventure', 'STILL READABLE'):
                assert anchor in recognized
            retained = [(r.text, r.lines.tolist()) for r in regions if 'evening' not in r.text]
            assert mask[top:top + 135].any()
            failure = 'empty-paragraph'
            regions, mask, raw, bubbles = engine.analyze(rgb)
            recognized = ' '.join(region.text for region in regions)
            assert 'evening' not in recognized and 'STILL READABLE' in recognized
            assert [(r.text, r.lines.tolist()) for r in regions] == retained
            cleaned = engine.inpaint(rgb, mask, raw, bubbles, [serialize_region(r) for r in regions])
            assert cleaned.shape == rgb.shape and cleaned.dtype == np.uint8
            assert np.any(cleaned != rgb)
            left, y, right, bottom = draw.textbbox((70, top + 90), sentences[1], font=font)
            assert not mask[y:bottom, left:right].any()
            np.testing.assert_array_equal(cleaned[y:bottom, left:right], rgb[y:bottom, left:right])
            failure = 'all-empty'
            assert engine.analyze(rgb) == ([], None, None, None)
    finally:
        engine.close()


@pytest.mark.parametrize(('filename', 'family'), [
    ('NotoNaskhArabic-Regular.ttf', 'Noto Naskh Arabic'),
    ('NotoSansArabic.ttf', 'Noto Sans Arabic'),
])
def test_arabic_native_shaping_bidi_and_diacritics_offline(assets, offline, filename, family):
    models, fonts = assets
    renderer = Renderer(models, fonts)
    from manga_translator.rendering.text_render import _fonts
    _fonts.set_font(renderer.font_by_name[filename])
    # Contextual Arabic forms must differ from isolated cmap glyphs. Some Noto
    # versions compose lam-alef from two joining glyphs rather than one ligature.
    _, _, layout, _ = _fonts._create_text_layout('سلام', 48)
    runs = layout.glyphRuns()
    assert len(runs) == 1 and runs[0].isRightToLeft()
    assert runs[0].rawFont().familyName() == family
    assert runs[0].glyphIndexes() != runs[0].rawFont().glyphIndexesForString('سلام')
    positions = [point.x() for point in runs[0].positions()]
    assert positions == sorted(positions, reverse=True)
    text = 'مُحَمَّد: لا لأ لإ لآ، عام 200 AMIRA (٢٠٠)؟…'
    assert renderer.covers(text)
    _, _, mixed, _ = _fonts._create_text_layout(text, 48)
    runs = mixed.glyphRuns()
    assert any(run.isRightToLeft() for run in runs)
    assert any(not run.isRightToLeft() for run in runs)
    assert all(index != 0 for run in runs for index in run.glyphIndexes())


@pytest.mark.parametrize(('language', 'text'), [
    ('en', 'THIS PERSON COULD WIELD MAGIC TO CONTROL ALL THINGS.'),
    ('ar', 'كان هذا الشخص قادرًا على تسخير السحر للتحكم في كل شيء.'),
    ('vi', 'NGƯỜI NÀY CÓ THỂ DÙNG PHÉP THUẬT ĐỂ ĐIỀU KHIỂN VẠN VẬT.'),
])
def test_translated_glyphs_stay_inside_bubble_mask(bubble_layout, language, text):
    renderer, observed, _ = bubble_layout
    from manga_translator.utils import TextBlock, erode_bubble_mask
    source = np.full((500, 600, 3), 255, dtype=np.uint8)
    block = TextBlock(np.array([[[230, 130], [350, 130], [350, 340], [230, 340]]]),
                      ['source'], font_size=56, direction='v')
    mask = np.zeros(source.shape[:2], dtype=np.uint8)
    mask[95:375, 180:410] = 255
    output = renderer.render(source, source, [serialize_region(block)], [text], language, mask)
    painted = np.any(output != source, axis=2)
    assert painted.any()
    assert not np.any(painted & (mask == 0))
    assert re.sub(r'\s*\[BR\]\s*', ' ', observed[0].translation).split() == text.split()
    assert observed[0].font_size < block.font_size
    # Line counts and nominal sizes vary with the selected font. Keep whole
    # words and verify the largest safe size, not the retired Bangers metrics.
    assert_maximum_scaled_font(observed[0], erode_bubble_mask(mask, .02))


def assert_maximum_scaled_font(block, mask):
    """Measure actual glyphs, not a font-specific historical pixel-size floor."""
    from mtu_engine.engine import configuration
    from manga_translator import rendering
    config = configuration()
    config._current_region = block
    rendering.text_render.set_font(block.font_family)
    assert rendering._polygon_fully_inside_mask(block.dst_points[0], mask)
    # The smallest layout font that could raise the final scaled size by 1px.
    next_size = math.ceil((block.font_size + 1) / config.render.font_scale_ratio)
    points = rendering._calc_region_dst_points_for_font(
        region=block, font_size=next_size, render_horizontally=block.horizontal,
        line_spacing_multiplier=block.line_spacing, letter_spacing_multiplier=block.letter_spacing,
        config=config, anchor_mode='center')
    assert not rendering._polygon_fully_inside_mask(points[0], mask)


@pytest.mark.parametrize('text', ['NEW', 'STOP!', 'سلام'])
def test_single_word_preserves_characters_without_internal_breaks(bubble_layout, text):
    renderer, observed, _ = bubble_layout
    from manga_translator.utils import TextBlock
    source = np.full((300, 300, 3), 255, dtype=np.uint8)
    block = TextBlock(np.array([[[100, 40], [190, 40], [190, 220], [100, 220]]]),
                      ['原'], font_size=90, direction='v')
    output = renderer.render(source, source, [serialize_region(block)], [text],
                             'ar' if text == 'سلام' else 'en', None)
    assert np.any(output != source)
    assert observed[0].translation == text
    assert observed[0].font_size > 0


@pytest.mark.parametrize(('language', 'text', 'shape'), [
    ('de', 'WUNDERSCHÖN.', 'solid'),
    ('en', 'Beautiful.', 'solid'),
    ('de', 'WUNDERSCHÖN.', 'hole'),
    ('de', 'WUNDERSCHÖN.', 'neighbor'),
])
@pytest.mark.parametrize('font_id', [None, 'patrick-hand'])
def test_horizontal_single_word_uses_enclosing_bubble_without_splitting(assets, offline, monkeypatch, language, text, shape, font_id):
    import cv2
    from mtu_engine import layout
    # This regression isolates the whole-word horizontal/rich path. Stacked
    # words may now span columns and have separate character-completeness tests.
    monkeypatch.setattr(layout, '_can_stack_text', lambda *_: False)
    models, fonts = assets
    renderer = Renderer(models, fonts)
    from manga_translator import rendering
    from manga_translator.utils import TextBlock, erode_bubble_mask
    source = np.full((1000, 500, 3), 255, dtype=np.uint8)
    # Actual narrow vertical source geometry; target words must fit the bubble,
    # not stay trapped inside the original 43px-wide OCR rectangle.
    block = TextBlock([[[207, 770], [250, 770], [250, 851], [207, 851]]],
                      ['source'], font_size=43, direction='v')
    region = serialize_region(block)
    mask = np.zeros(source.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, (230, 812), (86, 86), 0, 0, 360, 255, -1)
    if shape == 'hole':
        # Outside the source lines, but inside the larger word's possible box.
        mask[792:831, 280:296] = 0
    elif shape == 'neighbor':
        cv2.ellipse(mask, (415, 812), (60, 82), 0, 0, 360, 255, -1)
    dispatch, observed = rendering.dispatch, []

    async def capture(*args, **kwargs):
        result = await dispatch(*args, **kwargs)
        observed.extend((str(args[2].render.layout_mode), item) for item in args[1])
        return result

    monkeypatch.setattr(rendering, 'dispatch', capture)
    free_output = renderer.render(source, source, [region], [text], language, None, font_id=font_id)
    free_mode, free = observed.pop()
    assert free_mode == 'smart_scaling' and free.translation == text
    assert np.any(free_output != source) and 0 < free.font_size < 20
    output = renderer.render(source, source, [region], [text], language, mask, font_id=font_id)
    mode, enclosed = observed.pop()
    painted = np.any(output != source, axis=2)
    assert mode == 'balloon_fill' and painted.any()
    assert enclosed.horizontal and enclosed.translation == text and '[BR]' not in enclosed.translation
    assert not np.any(painted & (mask == 0))
    assert enclosed.font_size > 0
    assert enclosed.font_size > free.font_size
    assert_maximum_scaled_font(enclosed, erode_bubble_mask(mask, .02))
    if shape == 'neighbor':
        assert not np.any(painted[:, 355:])
    # No per-page layout state may turn a later uncontained word into a bubble.
    assert np.array_equal(renderer.render(source, source, [region], [text], language, None, font_id=font_id), free_output)


@pytest.mark.parametrize(('texts', 'lefts', 'shape'), [
    (['WUNDERSCHÖN.', 'WUNDERSCHÖN.'], (158, 398), 'rectangle'),
    (['WUNDERSCHÖN.', 'DIE WELT IST SCHÖN.'], (158, 370), 'rectangle'),
    (['DIE WELT IST SCHÖN.', 'WUNDERSCHÖN.'], (158, 370), 'rectangle'),
    (['WUNDERSCHÖN.', 'WUNDERSCHÖN.'], (240, 298), 'ellipse'),
], ids=['rich-rich', 'rich-plain', 'plain-rich', 'nearby-anchors'])
def test_enlarged_horizontal_single_word_avoids_other_paragraph_in_same_bubble(bubble_layout, monkeypatch, texts, lefts, shape):
    import cv2
    from mtu_engine import layout
    # Preserve this test's original rich-word collision coverage independently
    # of the additional vertical alternatives exercised by shared-bubble tests.
    monkeypatch.setattr(layout, '_can_stack_text', lambda *_: False)
    renderer, observed, rendering = bubble_layout
    from manga_translator.utils import TextBlock
    source = np.full((400, 600, 3), 255, dtype=np.uint8)
    mask = np.zeros(source.shape[:2], dtype=np.uint8)
    if shape == 'ellipse':
        cv2.ellipse(mask, (300, 200), (250, 170), 0, 0, 360, 255, -1)
    else:
        mask[60:340, 40:560] = 255
    regions = [serialize_region(TextBlock([[[left, 160], [left + 43, 160],
        [left + 43, 241], [left, 241]]], ['source'], font_size=43, direction='v'))
        for left in lefts]
    output = renderer.render(source, source, regions, texts, 'de', mask)
    painted = np.any(output != source, axis=2)
    assert painted.any() and not np.any(painted & (mask == 0))
    assert len(observed) == 2
    for block, text in zip(observed, texts):
        assert block.horizontal and block.font_size > 0
        assert re.sub(r'\s*\[BR\]\s*', ' ', block.translation).split() == text.split()
        if len(text.split()) == 1:
            assert block.translation == text
    assert not rendering._polygons_overlap(observed[0].dst_points, observed[1].dst_points)


@pytest.mark.parametrize('angle', [0, 15, -15])
def test_long_translation_near_page_edge_preserves_all_words(bubble_layout, angle):
    renderer, observed, _ = bubble_layout
    from manga_translator.utils import TextBlock
    block = TextBlock(np.array([[[470, 60], [570, 60], [570, 300], [470, 300]]]),
                      ['source'], font_size=48, angle=angle, direction='h')
    source = np.full((400, 600, 3), 255, dtype=np.uint8)
    text = 'KEINDAHAN DUNIA YANG SESUNGGUHNYA.'
    output = renderer.render(source, source, [serialize_region(block)], [text], 'id', None)
    assert np.any(output != source)
    points = observed[0].dst_points.reshape(-1, 2)
    assert points[:, 0].min() >= 1 and points[:, 0].max() <= 599
    assert points[:, 1].min() >= 1 and points[:, 1].max() <= 399
    assert re.sub(r'\s*\[BR\]\s*', ' ', observed[0].translation).split() == text.split()


@pytest.fixture
def bubble_layout(assets, offline, monkeypatch):
    """Real Qt pixels and final native boxes; no OCR or model inference."""
    renderer = Renderer(*assets)
    from manga_translator import rendering
    dispatch, observed = rendering.dispatch, []

    async def capture(*args, **kwargs):
        result = await dispatch(*args, **kwargs)
        observed.extend(args[1])
        return result

    monkeypatch.setattr(rendering, 'dispatch', capture)
    return renderer, observed, rendering


@pytest.mark.parametrize('language,text,direction', [
    ('zh-Hans', '好呀', 'h'), ('ja', 'そうだね', 'v'), ('en', 'Yes!', 'h'),
    ('en', 'Here we go!', 'h'), ('en', 'Absolutely.', 'h'), ('ar', 'أجل، هيا بنا', 'h'),
    ('en', 'All[BR]right!', 'h'),
])
def test_short_translation_scales_with_room_at_the_same_source_size(bubble_layout, language, text, direction):
    import cv2
    from manga_translator.utils import TextBlock, erode_bubble_mask
    renderer, observed, native = bubble_layout
    image = np.full((320, 480, 3), 255, np.uint8)
    region = serialize_region(TextBlock([[[220, 150], [260, 150], [260, 170], [220, 170]]],
        ['a longer source paragraph printed in small type'], font_size=12, direction=direction))
    sizes = []
    for axes in ((46, 30), (170, 105), (225, 145)):
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.ellipse(mask, (240, 160), axes, 0, 0, 360, 255, -1)
        rendered = renderer.render(image, image, [region], [text], language, mask)
        block = observed.pop()
        sizes.append(block.font_size)
        assert block.original_font_size == region['font_size'] and block.direction == direction
        np.testing.assert_array_equal(block.lines, region['lines'])
        if language in ('ja', 'zh-Hans'):
            assert block.translation.replace('[BR]', '') == text
        elif '[BR]' in text:
            assert block.translation == text
        else:
            assert block.translation.replace('[BR]', ' ').split() == text.split()
        inset = erode_bubble_mask(mask, .02)
        assert native._polygon_fully_inside_mask(block.dst_points[0], inset)
        ink = np.any(rendered != image, axis=2)
        assert ink.any() and not np.any(ink & (inset == 0))
        if axes[0] >= 170:
            assert block.translation == text  # Ample room: no extra line or column needed.
    assert sizes[0] <= sizes[1] == sizes[2]
    assert 12 < sizes[1] <= 15
    if text == 'Absolutely.':
        assert sizes[0] < sizes[1]  # The narrow bubble cannot afford the same increase.


@pytest.mark.parametrize('overlapping_boxes', [False, True])
@pytest.mark.parametrize('language,text', [('en', 'Yes!'), ('zh-Hans', '好呀')])
def test_free_short_word_grows_within_its_source_box(bubble_layout, overlapping_boxes, language, text):
    from manga_translator.utils import TextBlock
    renderer, observed, native = bubble_layout
    image = np.full((200, 260, 3), 255, np.uint8)
    region = serialize_region(TextBlock([[[35, 45], [215, 45], [215, 130], [35, 130]]],
        ['several lines of small original text'], font_size=12, direction='h'))
    regions, texts = [region], [text]
    if overlapping_boxes:
        regions.append(serialize_region(TextBlock([[[85, 65], [235, 65], [235, 110], [85, 110]]],
            ['another paragraph'], font_size=12, direction='h')))
        texts.append('Here!' if language == 'en' else '在这')
    rendered = renderer.render(image, image, regions, texts, language, None)
    block = observed[0]
    assert 12 < block.font_size <= 15 and block.translation == text
    if overlapping_boxes:
        assert observed[1].translation == texts[1] and observed[1].font_size <= 15
        assert not native._polygons_overlap(block.dst_points, observed[1].dst_points)
    ink_y, ink_x = np.where(np.any(rendered != image, axis=2))
    assert 35 <= ink_x.min() <= ink_x.max() <= regions[-1]['lines'][0][1][0]
    assert 45 <= ink_y.min() <= ink_y.max() <= 130


def test_roomy_balloon_keeps_the_source_type_hierarchy(bubble_layout):
    from manga_translator.utils import TextBlock
    renderer, observed, _ = bubble_layout
    image = np.full((220, 480, 3), 255, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[20:200, 20:460] = 255
    for source_size in (12, 48):
        region = serialize_region(TextBlock([[[160, 70], [320, 70], [320, 150], [160, 150]]],
            ['source'], font_size=source_size, direction='h'))
        for space in (mask, None):
            renderer.render(image, image, [region], ['好'], 'zh-Hans', space)
            size = observed.pop().font_size
            assert source_size <= size <= source_size + 4


def test_break_only_reply_leaves_the_cleaned_page_visible(bubble_layout):
    from manga_translator.utils import TextBlock
    renderer, _, _ = bubble_layout
    image = np.full((100, 140, 3), 255, np.uint8)
    region = serialize_region(TextBlock([[[35, 35], [95, 35], [95, 65], [35, 65]]],
        ['source'], font_size=12, direction='h'))
    for mask in (None, np.full(image.shape[:2], 255, np.uint8)):
        np.testing.assert_array_equal(renderer.render(image, image, [region], ['[BR]'], 'en', mask), image)


def test_roomy_bubble_can_move_a_short_reply_off_the_source_edge(bubble_layout, monkeypatch):
    from manga_translator.utils import TextBlock, erode_bubble_mask
    renderer, observed, native = bubble_layout
    image = np.full((180, 520, 3), 255, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[20:160, 20:500] = 255
    region = serialize_region(TextBlock([[[26, 75], [48, 75], [48, 95], [26, 95]]],
        ['small source near the balloon edge'], font_size=12, direction='h'))
    monkeypatch.setattr(native, 'find_largest_inscribed_rect',
                        lambda *_: pytest.fail('A plain bubble should not need a rectangle scan'))
    result = renderer.render(image, image, [region], ['好吧'], 'zh-Hans', mask)
    block = observed.pop()
    assert 12 < block.font_size <= 15
    np.testing.assert_allclose(block.center, (260, 90), atol=1)
    np.testing.assert_array_equal(block.lines, region['lines'])
    ink = np.any(image != result, axis=2)
    assert ink.any() and not np.any(ink & (erode_bubble_mask(mask, .02) == 0))


@pytest.mark.parametrize('edge', ['left', 'right'])
@pytest.mark.parametrize('cache_bytes', [0, None])
@pytest.mark.parametrize('font_id', [None, 'noto-sans'])
def test_narrow_caption_recovers_font_with_bounded_native_hyphenation(
        bubble_layout, monkeypatch, edge, cache_bytes, font_id):
    from manga_translator.utils import TextBlock, erode_bubble_mask
    renderer, observed, native = bubble_layout
    # Isolate dictionary-based horizontal recovery: a valid vertical
    # alternative can otherwise beat both horizontal sides of this A/B. Keep
    # the real Unicode safety decision shared by hyphenation and stacking.
    resize = native.resize_regions_to_font_size

    def reject_vertical_candidate(*args, **kwargs):
        points = resize(*args, **kwargs)
        for block in args[1]:
            if block.vertical:
                block.font_size = 1
        return points

    monkeypatch.setattr(native, 'resize_regions_to_font_size', reject_vertical_candidate)
    image = np.full((400, 300, 3), 255, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    x = 0 if edge == 'left' else 260
    mask[10:390, x:x+40] = 255
    region = serialize_region(TextBlock([[[x+8, 30], [x+30, 30], [x+30, 370], [x+8, 370]]],
        ['source'], font_size=24, direction='v'))
    snapshot = json.loads(json.dumps(region))
    text = 'Extraordinary circumstances / Please proceed carefully.'
    with monkeypatch.context() as baseline:
        baseline.setattr(native.text_render, 'select_hyphenator', lambda *_: None)
        renderer.render(image, image, [region], [text], 'en', mask,
                        mask_cache_bytes=cache_bytes, font_id=font_id)
    before = observed.pop()
    calls, rectangle = [], native.find_largest_inscribed_rect

    def counted(value):
        calls.append(value.shape)
        return rectangle(value)

    monkeypatch.setattr(native, 'find_largest_inscribed_rect', counted)
    result = renderer.render(image, image, [region], [text], 'en', mask,
                             mask_cache_bytes=cache_bytes, font_id=font_id)
    after = observed.pop()
    assert region == snapshot
    assert len(calls) == 1 and calls[0][1] < image.shape[1]  # One local alternative, never a page search.
    assert after.font_size > before.font_size
    assert '-[BR]' in after.translation
    assert re.sub(r'-\[BR\]', '', after.translation).replace('[BR]', ' ').split() == text.split()
    assert native._polygon_fully_inside_mask(after.dst_points[0], erode_bubble_mask(mask, .02))
    points = after.dst_points.reshape(-1, 2)
    assert (points >= (1, 1)).all() and (points <= (299, 399)).all()
    painted = np.any(result != image, axis=2)
    assert painted.any() and not np.any(painted & (erode_bubble_mask(mask, .02) == 0))
    np.testing.assert_array_equal(after.lines, np.asarray(region['lines']))


@pytest.mark.parametrize(('language', 'text', 'narrow'), [
    ('en', 'Extraordinary circumstances / Please proceed carefully.', False),
    ('en', 'Extraordinary[BR]circumstances / Please proceed carefully.', True),
    ('en', 'Extraordinary', True),
    ('ar', 'هذه ظروف استثنائية يرجى المتابعة بحذر', True),
])
def test_hyphenation_recovery_preserves_normal_explicit_single_and_unsupported_layouts(
        bubble_layout, monkeypatch, language, text, narrow):
    from manga_translator.utils import TextBlock
    renderer, observed, native = bubble_layout
    image = np.full((400, 300, 3), 255, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[10:390, 10:50 if narrow else 290] = 255
    # A wide bubble with a narrow source is now a recovery case too. Use
    # genuinely normal source geometry for the unchanged-layout control.
    lines = [[[18, 30], [42, 30], [42, 370], [18, 370]]] if narrow else [
        [[50, 110], [250, 110], [250, 290], [50, 290]]]
    region = serialize_region(TextBlock(lines, ['source'], font_size=24, direction='v'))
    with monkeypatch.context() as baseline:
        baseline.setattr(native.text_render, 'select_hyphenator', lambda *_: None)
        expected = renderer.render(image, image, [region], [text], language, mask)
    observed.clear()
    actual = renderer.render(image, image, [region], [text], language, mask)
    np.testing.assert_array_equal(actual, expected)
    assert '-[BR]' not in observed[-1].translation


def test_narrow_caption_recovery_reserves_other_paragraphs(bubble_layout):
    from manga_translator.utils import TextBlock, erode_bubble_mask
    renderer, observed, native = bubble_layout
    image = np.full((500, 240, 3), 255, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[10:490, 90:138] = 255
    regions = [serialize_region(TextBlock([[[102, y], [126, y], [126, y+100], [102, y+100]]],
        ['source'], font_size=24, direction='v')) for y in (50, 320)]
    texts = ['Extraordinary circumstances / Please proceed carefully.', 'Keep this paragraph.']
    result = renderer.render(image, image, regions, texts, 'en', mask)
    assert len(observed) == 2
    assert not native._polygons_overlap(observed[0].dst_points, observed[1].dst_points)
    for block, text in zip(observed, texts):
        if block.horizontal:
            assert re.sub(r'-\[BR\]', '', block.translation).replace('[BR]', ' ').split() == text.split()
        else:
            # Native stacked columns can continue a word in the next column.
            assert ''.join(block.translation.replace('[BR]', '').split()) == ''.join(text.split())
    painted = np.any(result != image, axis=2)
    assert painted.any() and not np.any(painted & (erode_bubble_mask(mask, .02) == 0))


def test_narrow_caption_keeps_existing_layout_when_candidate_is_not_larger(bubble_layout, monkeypatch):
    from manga_translator.utils import TextBlock
    from manga_translator.rendering import auto_linebreak
    renderer, observed, native = bubble_layout
    image = np.full((400, 200, 3), 255, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[10:390, 10:50] = 255
    region = serialize_region(TextBlock([[[18, 30], [42, 30], [42, 370], [18, 370]]],
        ['source'], font_size=24, direction='v'))
    text = 'Extraordinary circumstances / Please proceed carefully.'
    with monkeypatch.context() as baseline:
        baseline.setattr(native.text_render, 'select_hyphenator', lambda *_: None)
        expected = renderer.render(image, image, [region], [text], 'en', mask)
    calls = []

    def no_improvement(font_size, value, *args, **kwargs):
        calls.append(value)
        return [value], []

    monkeypatch.setattr(auto_linebreak, '_layout_horizontal_eng', no_improvement)
    actual = renderer.render(image, image, [region], [text], 'en', mask)
    assert calls == [text]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('shape', ['concave', 'hole', 'tail'])
@pytest.mark.parametrize('cache_bytes', [0, None])
def test_irregular_bubble_recovers_readable_layout_without_changing_source(bubble_layout, shape, cache_bytes):
    import cv2
    from mtu_engine.layout import associate_bubbles
    from manga_translator.utils import TextBlock, erode_bubble_mask
    renderer, observed, native = bubble_layout
    image = np.full((400, 500, 3), 255, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    if shape == 'concave':
        mask[30:350, 20:210] = 255
        mask[230:370, 20:430] = 255
        lines = [[[25, 70], [190, 70], [190, 100], [25, 100]],
                 [[25, 160], [190, 160], [190, 190], [25, 190]],
                 [[25, 250], [405, 250], [405, 285], [25, 285]]]
    elif shape == 'hole':
        mask[20:380, 20:450] = 255
        mask[140:240, 180:270] = 0
        lines = [[[40, 80], [165, 80], [165, 110], [40, 110]],
                 [[280, 260], [420, 260], [420, 290], [280, 290]]]
    else:
        mask[60:330, 140:450] = 255
        cv2.fillPoly(mask, [np.array([[20, 100], [160, 100], [160, 180]])], 255)
        lines = [[[70, 110], [120, 110], [120, 120], [70, 120]],
                 [[160, 240], [410, 240], [410, 270], [160, 270]]]
    block = TextBlock(lines, ['source'], font_size=26, direction='h', angle=1)
    associate_bubbles([block], mask)
    assert block.bubble_enclosed
    region = serialize_region(block)
    snapshot = json.loads(json.dumps(region))
    text = '虽然很厉害但是我不想学一般人会选择更容易掌握的招式'
    calls, rectangle = [], native.find_largest_inscribed_rect

    def counted(value):
        calls.append(value.shape)
        return rectangle(value)

    native.find_largest_inscribed_rect = counted
    try:
        output = renderer.render(image, image, [region], [text], 'zh-Hans', mask,
                                 mask_cache_bytes=cache_bytes)
    finally:
        native.find_largest_inscribed_rect = rectangle
    assert region == snapshot
    assert len(calls) == 1 and calls[0][0] < image.shape[0]  # One ROI-only alternative.
    painted = np.any(output != image, axis=2)
    inset = erode_bubble_mask(mask, .02)
    assert painted.any() and not np.any(painted & (inset == 0))
    assert observed[-1].font_size >= 16
    assert observed[-1].translation.replace('[BR]', '') == text
    np.testing.assert_array_equal(observed[-1].lines, block.lines)
    # The checkpoint carries ownership across inpaint/cache/IPC. Cached inset
    # and cold raw-mask paths must produce identical pixels and source geometry.
    _, labels, stats, _ = cv2.connectedComponentsWithStats(inset, connectivity=8)
    restored = renderer.render(image, image, [snapshot], [text], 'zh-Hans', inset,
        bubble_prepared=True, bubble_components=(labels, stats), mask_cache_bytes=cache_bytes)
    np.testing.assert_array_equal(restored, output)


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('language,texts', [
    ('zh-Hans', ['这是一段需要重新排版的文字不能挤到旁边', '中间这段文字也要保留', '最后一段同样需要空间']),
    ('en', ['This paragraph needs room without covering its neighbors.',
            'Keep the middle paragraph intact.', 'The final paragraph also needs space.']),
])
def test_shared_bubble_reserves_pending_and_placed_paragraphs(bubble_layout, reverse, language, texts):
    import cv2
    from manga_translator.utils import TextBlock, erode_bubble_mask
    renderer, observed, native = bubble_layout
    image = np.full((420, 600, 3), 255, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.ellipse(mask, (300, 210), (275, 180), 0, 0, 360, 255, -1)
    lines = [
        [[[75, 125], [130, 125], [130, 145], [75, 145]],
         [[45, 205], [140, 205], [140, 225], [45, 225]]],
        [[[270, 160], [330, 160], [330, 230], [270, 230]]],
        [[[440, 160], [520, 160], [520, 230], [440, 230]]],
    ]
    regions = [serialize_region(TextBlock(value, ['source'], font_size=26, direction='h')) for value in lines]
    if reverse:
        regions, texts = regions[::-1], texts[::-1]
    output = renderer.render(image, image, regions, texts, language, mask)
    assert len(observed) == 3
    for i, block in enumerate(observed):
        assert block.font_size > 0
        # Narrow paragraphs in a wide shared bubble can now use native
        # hyphenation too; reconstruct words before checking completeness.
        restored = block.translation.replace('-[BR]', '').replace('[BR]', '')
        assert ''.join(restored.split()) == ''.join(texts[i].split())
        for other in observed[i+1:]:
            assert not native._polygons_overlap(block.dst_points, other.dst_points)
    painted = np.any(output != image, axis=2)
    assert painted.any() and not np.any(painted & (erode_bubble_mask(mask, .02) == 0))


@pytest.mark.parametrize('edge,box', [
    ('top', (190, 0, 290, 38)), ('bottom', (190, 360, 290, 399)),
    ('left', (0, 180, 80, 220)), ('right', (419, 180, 499, 220)),
])
@pytest.mark.parametrize('angle', [0, 18])
def test_bubble_layout_checks_unclipped_page_edges(bubble_layout, edge, box, angle):
    from manga_translator.utils import TextBlock
    renderer, observed, _ = bubble_layout
    image = np.full((400, 500, 3), 255, np.uint8)
    mask = np.full(image.shape[:2], 255, np.uint8)
    x1, y1, x2, y2 = box
    block = TextBlock([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]],
                      ['source'], font_size=30, angle=angle, direction='h')
    text = '页面边缘的文字也应该完整保留不能被裁切或缩成看不清的小字'
    output = renderer.render(image, image, [serialize_region(block)], [text], 'zh-Hans', mask)
    points = observed[-1].dst_points.reshape(-1, 2)
    assert (points >= (1, 1)).all() and (points <= (499, 399)).all(), edge
    assert observed[-1].font_size >= 16
    assert observed[-1].translation.replace('[BR]', '') == text
    assert np.any(output != image)
    assert np.array_equal(output[[0, -1]], image[[0, -1]])
    assert np.array_equal(output[:, [0, -1]], image[:, [0, -1]])


@pytest.mark.parametrize('cache_bytes', [0, None])
def test_short_wide_bubble_shrinks_overflow_without_a_collision(bubble_layout, monkeypatch, cache_bytes):
    from manga_translator.utils import TextBlock
    renderer, observed, native = bubble_layout
    image = np.full((200, 1433, 3), 255, np.uint8)
    inset = np.zeros(image.shape[:2], np.uint8)
    inset[45:127, 420:1071] = 255
    region = {**serialize_region(TextBlock(
        [[[458, 60], [1056, 60], [1056, 110], [458, 110]]],
        ['source'], font_size=50, direction='h')), 'bubble_enclosed': True}
    snapshot = json.loads(json.dumps(region))
    text = 'AN EXTRAORDINARY ADVENTURE'
    resize = native.resize_regions_to_font_size

    def oversized(*args, **kwargs):
        points = resize(*args, **kwargs)
        # Reproduce a final native font that exceeds a short bubble after
        # wrapping, independently of platform-specific Qt font measurements.
        for index, block in enumerate(args[1]):
            block.font_size = 48
            args[2]._current_region = block
            native.text_render.set_font(block.font_family)
            points[index] = native._calc_region_dst_points_for_font(
                region=block, font_size=48, render_horizontally=True,
                line_spacing_multiplier=block.line_spacing,
                letter_spacing_multiplier=block.letter_spacing, config=args[2], anchor_mode='center')
        return points

    monkeypatch.setattr(native, 'resize_regions_to_font_size', oversized)
    output = renderer.render(image, image, [region], [text], 'en', inset,
        bubble_prepared=True, mask_cache_bytes=cache_bytes, font_id='comic-relief')
    assert region == snapshot and len(observed) == 1
    block = observed[0]
    assert 1 < block.font_size < 47
    assert block.translation.replace('[BR]', ' ').split() == text.split()
    assert native._polygon_fully_inside_mask(block.dst_points.reshape(-1, 2), inset)
    painted = np.any(output != image, axis=2)
    assert painted.any() and not np.any(painted & (inset == 0))
    np.testing.assert_array_equal(block.lines, region['lines'])


@pytest.mark.parametrize('language,text,passes', [
    ('zh-Hans', '正常气泡保持原有排版', 1), ('zh-Hans', '好呀', 0),
    ('en', 'Yes!', 1), ('en', 'Stop[BR]here!', 1)])
def test_safe_bubble_layout_does_not_search_alternate_rectangles(bubble_layout, monkeypatch, language, text, passes):
    from manga_translator.utils import TextBlock
    renderer, observed, native = bubble_layout
    image = np.full((300, 400, 3), 255, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[20:280, 20:380] = 255
    block = TextBlock([[[100, 90], [300, 90], [300, 200], [100, 200]]], ['source'],
                      font_size=26, direction='h')
    monkeypatch.setattr(native, 'find_largest_inscribed_rect',
                        lambda *_: pytest.fail('This layout should not need a rectangle scan'))
    layouts, layout = [], native._layout_regions_to_font_size

    def counted(*args, **kwargs):
        if args[1]:
            layouts.append(len(args[1]))
        return layout(*args, **kwargs)

    monkeypatch.setattr(native, '_layout_regions_to_font_size', counted)
    result = renderer.render(image, image, [serialize_region(block)], [text], language, mask)
    assert np.any(result != image) and len(observed) == 1
    assert layouts == [1] * passes  # Paint the validated geometry without repeating Qt layout.


@pytest.mark.parametrize('mask_kind', ['missing', 'empty', 'detached', 'tiny'])
def test_unusable_bubble_masks_keep_text_in_every_language_and_process(bubble_layout, assets, mask_kind):
    from classic_node.render_pool import RenderPool, render_page
    from manga_translator.utils import TextBlock
    renderer, observed, _ = bubble_layout
    image = np.full((240, 420, 3), 255, np.uint8)
    block = TextBlock([[[40, 40], [280, 40], [280, 200], [40, 200]]],
                      ['source'], font_size=24, direction='h')
    region = {**serialize_region(block), 'bubble_enclosed': True}
    snapshot = json.dumps(region, sort_keys=True)
    mask = None if mask_kind == 'missing' else np.zeros(image.shape[:2], np.uint8)
    if mask_kind == 'detached':
        mask[20:60, 340:400] = 255
    elif mask_kind == 'tiny':
        mask[100:102, 100:102] = 255
    options = {'alpha': None, 'version': 'bubble-fixture', 'analysis': {'input_hash': '0' * 64},
               'translated': {'analysis_hash': '1' * 64, 'revision': '2' * 64}, 'bubble_prepared': True}
    pool = RenderPool(*assets, workers=2, threads=1)
    try:
        for language, text in LANGUAGE_PROBES.items():
            observed.clear()
            args = (image, image, [region], [text], language, mask)
            expected = render_page(renderer, *args, **options)
            assert expected['result']['representation'] != 'original', language
            assert len(observed) == 1, language
            assert observed[0].font_size > 1, language
            if observed[0].font_size > region['font_size']:
                corners = observed[0].dst_points.reshape(-1, 2)
                assert (corners >= (40, 40)).all() and (corners <= (280, 200)).all(), language
            actual = re.sub(r'\s+', '', observed[0].translation.replace('[BR]', ''))
            assert actual == re.sub(r'\s+', '', text), language
            np.testing.assert_array_equal(observed[0].lines, region['lines'])
            assert pool.render(*args, **options) == expected, language
        assert json.dumps(region, sort_keys=True) == snapshot
        assert np.all(image == 255)
    finally:
        pool.close()


@pytest.mark.parametrize('language', ['zh-Hans', 'en', 'ko'])
def test_recovered_bubble_layout_matches_real_render_process_pool(assets, offline, language):
    import cv2
    from classic_node.render_pool import RenderPool, render_page
    from mtu_engine.layout import associate_bubbles
    renderer = Renderer(*assets)
    from manga_translator.utils import TextBlock, erode_bubble_mask
    image = np.full((360, 480, 3), 255, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    if language in ('en', 'ko'):
        mask[20:340, 20:60] = 255
        lines = [[[28, 40], [52, 40], [52, 320], [28, 320]]]
        text = ['개구쟁이 초등학생이냐고' if language == 'ko' else
                'Extraordinary circumstances / Please proceed carefully.']
    else:
        mask[20:330, 20:200] = 255
        mask[220:340, 20:450] = 255
        lines = [[[24, 70], [190, 70], [190, 100], [24, 100]],
                 [[30, 240], [430, 240], [430, 270], [30, 270]]]
        text = ['异形气泡在子进程中同样保持安全排版和可读字号']
    block = TextBlock(lines, ['source'], font_size=26, direction='h')
    associate_bubbles([block], mask)
    assert block.bubble_enclosed
    regions = [serialize_region(block)]
    inset = erode_bubble_mask(mask, .02)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(inset, connectivity=8)
    options = {'alpha': None, 'version': 'bubble-fixture', 'analysis': {'input_hash': '0' * 64},
               'translated': {'analysis_hash': '1' * 64, 'revision': '2' * 64},
               'bubble_prepared': True, 'bubble_components': (labels, stats)}
    expected = render_page(renderer, image, image, regions, text, language, inset, **options)
    pool = RenderPool(*assets, workers=2, threads=1)
    try:
        pool.warmup()
        assert pool.render(image, image, regions, text, language, inset, **options) == expected
    finally:
        pool.close()


@pytest.mark.parametrize('cache_bytes', [0, None])
@pytest.mark.parametrize(('language', 'text'), [
    ('ko', '실례할게'),
    ('ko', '개구쟁이 초등학생이냐고'),
    ('en', 'Extraordinary'),
    ('de', 'WUNDERSCHÖN.'),
    ('ru', 'Замечательно!'),
    ('id', 'Mengagumkan!'),
])
def test_narrow_translation_recovers_native_vertical_font(
        bubble_layout, monkeypatch, cache_bytes, language, text):
    from mtu_engine import layout
    from manga_translator.utils import TextBlock, erode_bubble_mask
    renderer, observed, native = bubble_layout
    image = np.full((440, 200, 3), 255, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[30:410, 78:122] = 255
    region = serialize_region(TextBlock([[[86, 45], [114, 45], [114, 395], [86, 395]]],
        ['source'], font_size=28, direction='v'))
    snapshot = json.loads(json.dumps(region))
    with monkeypatch.context() as baseline:
        baseline.setattr(layout, '_can_stack_text', lambda *_: False)
        renderer.render(image, image, [region], [text], language, mask,
                        mask_cache_bytes=cache_bytes)
    before = observed.pop()
    calls, rectangle = [], native.find_largest_inscribed_rect

    def counted(value):
        calls.append(value.shape)
        return rectangle(value)

    monkeypatch.setattr(native, 'find_largest_inscribed_rect', counted)
    result = renderer.render(image, image, [region], [text], language, mask,
                             mask_cache_bytes=cache_bytes)
    after = observed.pop()
    assert region == snapshot
    assert before.horizontal and after.vertical
    assert after.font_size > before.font_size
    assert after.font_size >= region['font_size'] * .5
    assert after.original_font_size == region['font_size']
    assert len(calls) == 1 and calls[0][1] < image.shape[1]
    assert ''.join(after.translation.replace('[BR]', '').split()) == ''.join(text.split())
    np.testing.assert_array_equal(after.lines, np.asarray(region['lines']))
    inset = erode_bubble_mask(mask, .02)
    assert native._polygon_fully_inside_mask(after.dst_points[0], inset)
    painted = np.any(result != image, axis=2)
    assert painted.any() and not np.any(painted & (inset == 0))


def test_narrow_source_recovers_inside_wide_shared_bubble(bubble_layout, monkeypatch):
    from manga_translator.utils import TextBlock, erode_bubble_mask
    renderer, observed, native = bubble_layout
    image = np.full((440, 400, 3), 255, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[20:220, 20:380] = 255
    mask[180:420, 40:84] = 255
    regions = [serialize_region(TextBlock([[[46, 250], [76, 250], [76, 390], [46, 390]]],
        ['source'], font_size=28, direction='v')),
        serialize_region(TextBlock([[[130, 65], [310, 65], [310, 165], [130, 165]]],
        ['source'], font_size=28, direction='h'))]
    texts = ['개구쟁이 초등학생이냐고', '이 문단의 글자는 다른 문단이 가리면 안 됩니다.']
    snapshot = json.loads(json.dumps(regions))
    calls, rectangle = [], native.find_largest_inscribed_rect

    def counted(value):
        calls.append(value.shape)
        return rectangle(value)

    monkeypatch.setattr(native, 'find_largest_inscribed_rect', counted)
    result = renderer.render(image, image, regions, texts, 'ko', mask)
    assert regions == snapshot and len(observed) == 2
    assert observed[0].font_size >= regions[0]['font_size'] * .5
    assert not native._polygons_overlap(observed[0].dst_points, observed[1].dst_points)
    # Native initial mask layout can receive the page mask; the additional
    # recovery still searches only one component-local rectangle.
    local_calls = [shape for shape in calls if shape != image.shape[:2]]
    assert len(local_calls) == 1 and local_calls[0][1] < image.shape[1]
    for block, text in zip(observed, texts):
        assert ''.join(block.translation.replace('[BR]', '').split()) == ''.join(text.split())
    painted = np.any(result != image, axis=2)
    assert painted.any() and not np.any(painted & (erode_bubble_mask(mask, .02) == 0))


@pytest.mark.parametrize(('language', 'text', 'narrow'), [
    ('ko', '내일 제출인 레포트가 안 끝나아~!!', False),
    ('ko', '개구쟁이[BR]초등학생이냐고', True),
    ('en', 'Extraordinary[BR]circumstances', True),
    ('ar', 'استثنائية', True),
    ('en', 'Magicسلام', True),
    ('en', 'E\u0301xtraordinaire', True),
    ('ko', '\u1100\u1162구쟁이', True),
])
def test_vertical_recovery_preserves_normal_explicit_and_shaped_text(
        bubble_layout, monkeypatch, language, text, narrow):
    from mtu_engine import layout
    from manga_translator.utils import TextBlock
    renderer, observed, _ = bubble_layout
    image = np.full((440, 260, 3), 255, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    left, right = (108, 152) if narrow else (25, 235)
    mask[30:410, left:right] = 255
    region = serialize_region(TextBlock([[[116, 45], [144, 45], [144, 395], [116, 395]]],
        ['source'], font_size=28, direction='v'))
    with monkeypatch.context() as baseline:
        baseline.setattr(layout, '_can_stack_text', lambda *_: False)
        expected = renderer.render(image, image, [region], [text], language, mask)
    observed.clear()
    actual = renderer.render(image, image, [region], [text], language, mask)
    np.testing.assert_array_equal(actual, expected)
    assert observed[-1].horizontal


@pytest.mark.parametrize('shape', ['left-edge', 'hole'])
def test_vertical_recovery_keeps_page_edges_and_mask_holes(bubble_layout, shape):
    from manga_translator.utils import TextBlock, erode_bubble_mask
    renderer, observed, native = bubble_layout
    image = np.full((440, 200, 3), 255, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    left = 0 if shape == 'left-edge' else 78
    mask[10:430, left:left+44] = 255
    if shape == 'hole':
        mask[205:225, left:left+17] = 0
    region = serialize_region(TextBlock([[[left+20, 30], [left+40, 30],
        [left+40, 410], [left+20, 410]]], ['source'], font_size=28, direction='v'))
    result = renderer.render(image, image, [region], ['실례할게'], 'ko', mask)
    block = observed[-1]
    assert block.font_size >= 14
    points = block.dst_points.reshape(-1, 2)
    assert (points >= (1, 1)).all() and (points <= (199, 439)).all()
    inset = erode_bubble_mask(mask, .02)
    assert native._polygon_fully_inside_mask(block.dst_points[0], inset)
    painted = np.any(result != image, axis=2)
    assert painted.any() and not np.any(painted & (inset == 0))


def test_vertical_recovery_rejects_smaller_candidate_and_restores_native_options(
        bubble_layout, monkeypatch):
    from mtu_engine import layout
    from manga_translator.utils import TextBlock
    renderer, observed, native = bubble_layout
    image = np.full((440, 200, 3), 255, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[30:410, 78:122] = 255
    region = serialize_region(TextBlock([[[86, 45], [114, 45], [114, 395], [86, 395]]],
        ['source'], font_size=28, direction='v'))
    with monkeypatch.context() as baseline:
        baseline.setattr(layout, '_can_stack_text', lambda *_: False)
        expected = renderer.render(image, image, [region], ['실례할게'], 'ko', mask)
    observed.clear()
    calls, resize, dispatch = [], native.resize_regions_to_font_size, native.dispatch

    def smaller_candidate(*args, **kwargs):
        points = resize(*args, **kwargs)
        for block in args[1]:
            if block.vertical:
                assert not args[2].render.bubble_layout_english
                calls.append(block)
                block.font_size = 1
        return points

    async def check_restored_options(*args, **kwargs):
        assert args[2].render.bubble_layout_english
        return await dispatch(*args, **kwargs)

    monkeypatch.setattr(native, 'resize_regions_to_font_size', smaller_candidate)
    monkeypatch.setattr(native, 'dispatch', check_restored_options)
    actual = renderer.render(image, image, [region], ['실례할게'], 'ko', mask)
    np.testing.assert_array_equal(actual, expected)
    assert len(calls) == 1 and observed[-1].horizontal


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('source_sizes', [(28, 28), (20, 36), (12, 12)])
@pytest.mark.parametrize(('language', 'texts'), [
    ('ko', ['놀라워!', '서두르지 마, 내가 도와줄게.']),
    ('en', ['Incredible!', "Don't panic. I'll help you."]),
    ('fr', ['Incroyable !', "Pas de panique, je vais t'aider."]),
    ('ar', ['مذهل!', 'لا تقلق، سأساعدك.']),
])
def test_shared_bubble_growth_leaves_room_for_both_paragraphs(
        bubble_layout, monkeypatch, reverse, source_sizes, language, texts):
    from mtu_engine import layout
    from manga_translator.utils import TextBlock, erode_bubble_mask
    renderer, observed, native = bubble_layout
    image = np.full((440, 400, 3), 255, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[20:220, 20:380] = 255
    mask[180:420, 40:84] = 255
    regions = [serialize_region(TextBlock([[[46, 250], [76, 250], [76, 390], [46, 390]]],
        ['source'], font_size=source_sizes[0], direction='v')),
        serialize_region(TextBlock([[[130, 65], [310, 65], [310, 165], [130, 165]]],
        ['source'], font_size=source_sizes[1], direction='h'))]
    if reverse:
        regions, texts = regions[::-1], texts[::-1]
    snapshot = json.loads(json.dumps(regions))
    with monkeypatch.context() as reserved:
        reserved.setattr(layout, 'grow_into_space', lambda b, text, p, *args: (b, p))
        renderer.render(image, image, regions, texts, language, mask)
    reserved_sizes = [b.font_size for b in observed]
    observed.clear()
    result = renderer.render(image, image, regions, texts, language, mask)
    assert regions == snapshot and len(observed) == len(regions)
    if source_sizes == (12, 12):
        assert any(b.font_size > 12 for b in observed)
    inset = erode_bubble_mask(mask, .02)
    for block, text, source, reserved in zip(observed, texts, regions, reserved_sizes, strict=True):
        source_font = source['font_size']
        assert block.font_size >= max(.7 * source_font, reserved)
        assert block.original_font_size == source_font
        assert ''.join(re.sub(r'-\[BR\]', '', block.translation).replace('[BR]', '').split()) == ''.join(text.split())
        np.testing.assert_array_equal(block.lines, np.asarray(source['lines']))
        points = block.dst_points.reshape(-1, 2)
        assert np.isfinite(points).all()
        assert (points >= (1, 1)).all() and (points <= (399, 439)).all()
        assert native._polygon_fully_inside_mask(points, inset)
        if language == 'ar':
            assert block.horizontal  # Keep Arabic joins and logical character order intact.
    assert not native._polygons_overlap(observed[0].dst_points, observed[1].dst_points)
    painted = np.any(result != image, axis=2)
    assert painted.any() and not np.any(painted & (inset == 0))


@pytest.mark.parametrize('cache_bytes', [0, None])
@pytest.mark.parametrize('edge', [False, True])
@pytest.mark.parametrize(('language', 'text'), [
    ('ko', '내일 제출할 보고서를 아직 못 끝냈어!'),
    ('en', 'Pneumonoultramicroscopicsilicovolcanoconiosis!'),
    ('fr', "Je n'ai pas encore fini le rapport à rendre demain !"),
    ('es', '¡Todavía no he terminado el informe que tengo que entregar mañana!'),
    ('pt-BR', 'Ainda não terminei o relatório que preciso entregar amanhã!'),
    ('tr', 'Yarın teslim etmem gereken raporu hâlâ bitiremedim!'),
    ('ar', 'لم أنته بعد من التقرير الذي يجب تسليمه غدا!'),
])
def test_narrow_long_translation_uses_height_without_losing_larger_single_column(
        bubble_layout, monkeypatch, cache_bytes, edge, language, text):
    from types import SimpleNamespace
    from mtu_engine import layout
    from mtu_engine.engine import configuration
    from manga_translator.rendering import auto_linebreak
    from manga_translator.utils import TextBlock, erode_bubble_mask
    renderer, observed, native = bubble_layout
    image = np.full((440, 400, 3), 255, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    left = 0 if edge else 168
    mask[50:390, left:left+64] = 255
    if edge:
        mask[195:225, :17] = 0
    source_left = 25 if edge else 186
    region = serialize_region(TextBlock([[[source_left, 75], [source_left+28, 75],
        [source_left+28, 365], [source_left, 365]]], ['source'], font_size=28, direction='v'))
    snapshot = json.loads(json.dumps(region))
    solve = auto_linebreak.solve_no_br_layout

    def single_column_only(*args, **kwargs):
        config = kwargs.get('config')
        if config is not None and getattr(config, '_current_region', None) is None:
            # Disable only the new copy-local column alternative. Native
            # source-region layout remains real, including its one-column rule.
            assert not kwargs['horizontal']
            return SimpleNamespace(text_with_br=kwargs['text'])
        return solve(*args, **kwargs)

    with monkeypatch.context() as baseline:
        baseline.setattr(auto_linebreak, 'solve_no_br_layout', single_column_only)
        baseline.setattr(native.text_render, 'select_hyphenator', lambda *_: None)
        baseline.setattr(layout, '_wrap_words', lambda value, *_: value)
        renderer.render(image, image, [region], [text], language, mask,
                        mask_cache_bytes=cache_bytes)
    before = observed.pop()
    rectangles, candidates = [], []
    rectangle, resize = native.find_largest_inscribed_rect, native.resize_regions_to_font_size

    def counted_rectangle(value):
        rectangles.append(value.shape)
        return rectangle(value)

    def capture_candidates(*args, **kwargs):
        points = resize(*args, **kwargs)
        config = args[2]
        if config.render.layout_mode == 'balloon_fill' and not config.render.balloon_fill_mask_layout:
            if any(candidate.vertical for candidate in args[1]):
                assert not config.render.bubble_layout_english
            candidates.extend(args[1])
        return points

    monkeypatch.setattr(native, 'find_largest_inscribed_rect', counted_rectangle)
    monkeypatch.setattr(native, 'resize_regions_to_font_size', capture_candidates)
    output = renderer.render(image, image, [region], [text], language, mask,
                             mask_cache_bytes=cache_bytes)
    after = observed.pop()
    assert region == snapshot and after.original_font_size == region['font_size']
    np.testing.assert_array_equal(after.lines, np.asarray(region['lines']))
    if language == 'ko' and edge:
        # The pinned fixture's old single column is 18px; a more square
        # multi-column alternative must not replace it with a smaller size.
        assert before.vertical and before.font_size >= .5 * region['font_size']
        assert after.font_size >= before.font_size
    elif language == 'ar':
        # Mask shrinking also rescues the disabled-word-wrap control. Its
        # different breaks are no longer a smaller-font baseline; measure
        # the shaped, whole-word layout against its actual safe font budget.
        assert_maximum_scaled_font(after, erode_bubble_mask(mask, .02))
    else:
        assert after.font_size > before.font_size
    if language == 'ko' and not edge:
        assert before.vertical and '[BR]' not in before.translation
        assert after.vertical and '[BR]' in after.translation
    if len(text.split()) == 1:
        from manga_translator.rendering.rich_text import legacy_line_breaks_to_document
        # A long rich horizontal word can legitimately continue across native
        # vertical columns, but only BR may be inserted and direction stays v.
        assert before.vertical and '[BR]' not in before.translation
        assert after.vertical and '[BR]' in after.translation
        assert after.translation.replace('[BR]', '') == text
        # Native painting normalizes the new BR text to a document; it must
        # not retain the original unbroken horizontal rich paragraph.
        assert after.translation_rich == legacy_line_breaks_to_document(after.translation).to_dict()
    assert 1 <= len(candidates) <= 3  # h, original v, at most one reflowed v.
    assert len([shape for shape in rectangles if shape != image.shape[:2]]) == 1
    # These candidates reserve space; the final drawable type may grow afterwards.
    maximum = native._apply_final_font_constraints(region['font_size'], configuration())
    assert all(candidate.font_size <= maximum for candidate in candidates)
    restored = re.sub(r'-\[BR\]', '', after.translation).replace('[BR]', '')
    assert ''.join(restored.split()) == ''.join(text.split())
    if language == 'ar':
        assert after.horizontal and all(candidate.horizontal for candidate in candidates)
        assert after.translation.replace('[BR]', ' ').split() == text.split()
    points = after.dst_points.reshape(-1, 2)
    assert np.isfinite(points).all()
    assert (points >= (1, 1)).all() and (points <= (399, 439)).all()
    inset = erode_bubble_mask(mask, .02)
    assert native._polygon_fully_inside_mask(points, inset)
    painted = np.any(output != image, axis=2)
    assert painted.any() and not np.any(painted & (inset == 0))


@pytest.mark.parametrize(('text', 'whole_words'), [
    ('Astral 😀 characters remain intact.', False),
    ('E\u0301clair cafe\u0301 déjà vu', True),
    ('Family 👨\u200d👩\u200d👧\u200d👦 together', False),
    ('لم أنته بعد من التقرير الذي يجب تسليمه غدا!', True),
    ('A\u00a0B stay together', False),
])
def test_qt_word_recovery_preserves_utf16_graphemes_and_shaped_words(
        bubble_layout, text, whole_words):
    from PyQt6.QtCore import QTextBoundaryFinder
    from mtu_engine.layout import _wrap_words
    renderer, _, native = bubble_layout
    native.text_render.set_font(renderer.fonts[0])
    wrapped = _wrap_words(text, 28, 45, 1.0)
    assert '[BR]' in wrapped
    assert ''.join(wrapped.replace('[BR]', '').split()) == ''.join(text.split())
    boundaries = QTextBoundaryFinder(QTextBoundaryFinder.BoundaryType.Grapheme, text)
    position = 0
    for line in wrapped.split('[BR]'):
        start = text.index(line, position)
        assert not text[position:start].strip()
        position = start + len(line)
        for boundary in (start, position):
            boundaries.setPosition(len(text[:boundary].encode('utf-16-le')) // 2)
            assert boundaries.isAtBoundary()
    assert not text[position:].strip()
    if whole_words:
        assert wrapped.replace('[BR]', ' ').split() == text.split()
    if '\u00a0' in text:
        assert 'A\u00a0B' in wrapped  # A no-break space is not a permitted line boundary.


@pytest.mark.parametrize(('language', 'text'), [
    ('en', 'Extraordinary E\u0301clair circumstances'),
    ('fr', 'Extraordinaire A\u200dB maintenant'),
    ('en', 'Extraordinary سلام circumstances'),
])
def test_complex_text_recovery_uses_qt_words_not_codepoint_hyphenation(
        bubble_layout, monkeypatch, language, text):
    from mtu_engine import layout
    from manga_translator.rendering import auto_linebreak
    from manga_translator.utils import TextBlock, erode_bubble_mask
    renderer, observed, native = bubble_layout
    image = np.full((440, 260, 3), 255, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[30:410, 108:152] = 255
    region = serialize_region(TextBlock([[[116, 45], [144, 45], [144, 395], [116, 395]]],
        ['source'], font_size=28, direction='v'))
    calls, wrap = [], layout._wrap_words

    def record_wrap(value, *args):
        calls.append(value)
        return wrap(value, *args)

    monkeypatch.setattr(layout, '_wrap_words', record_wrap)
    monkeypatch.setattr(auto_linebreak, '_layout_horizontal_eng',
        lambda *args, **kwargs: pytest.fail('Complex text must not use codepoint hyphenation fallback'))
    output = renderer.render(image, image, [region], [text], language, mask)
    block = observed.pop()
    assert calls == [text]  # Recovery ran, rather than passing via the ordinary fast path.
    assert block.horizontal
    assert block.translation.replace('[BR]', ' ').split() == text.split()
    np.testing.assert_array_equal(block.lines, np.asarray(region['lines']))
    inset = erode_bubble_mask(mask, .02)
    assert native._polygon_fully_inside_mask(block.dst_points[0], inset)
    painted = np.any(output != image, axis=2)
    assert painted.any() and not np.any(painted & (inset == 0))


def _shared_reflow_page(source_sizes=(28, 19)):
    """A generic L-shaped bubble, not a private page or translated checkpoint."""
    from manga_translator.utils import TextBlock
    image = np.full((320, 280, 3), 255, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[25:250, 75:230] = 255
    mask[170:250, 25:220] = 255
    lines = [
        [[[130, 80], [215, 80], [215, 230], [130, 230]]],
        [[[40, 180], [59, 180], [59, 240], [40, 240]]],
    ]
    regions = [serialize_region(TextBlock(value, ['source'], font_size=size, direction='v'))
               for value, size in zip(lines, source_sizes)]
    texts = ['The library is open from nine until six.',
             'Ask the receptionist about the updated timetable.']
    return image, mask, regions, texts


def _assert_shared_reflow_pixels(image, output, mask, regions, texts, observed, native):
    from manga_translator.utils import erode_bubble_mask
    by_source = {tuple(np.asarray(block.lines).ravel()): block for block in observed}
    assert len(by_source) == len(regions)
    blocks = [by_source[tuple(np.asarray(region['lines']).ravel())] for region in regions]
    inset = erode_bubble_mask(mask, .02)
    for i, (block, region, text) in enumerate(zip(blocks, regions, texts)):
        assert block.original_font_size == region['font_size']
        assert block.font_size > 0
        restored = re.sub(r'-\[BR\]', '', block.translation).replace('[BR]', '')
        assert ''.join(restored.split()) == ''.join(text.replace('[BR]', '').split())
        np.testing.assert_array_equal(block.lines, np.asarray(region['lines']))
        vertices = block.dst_points.reshape(-1, 2)
        assert np.isfinite(vertices).all() and (vertices >= (1, 1)).all()
        assert (vertices <= (image.shape[1]-1, image.shape[0]-1)).all()
        assert native._polygon_fully_inside_mask(vertices, inset)
        assert all(not native._polygons_overlap(block.dst_points, other.dst_points)
                   for other in blocks[i+1:])
    painted = np.any(output != image, axis=2)
    assert painted.any() and not np.any(painted & (inset == 0))
    return blocks


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize(('language', 'texts', 'source_sizes'), [
    ('en', None, (28, 19)),
    ('en', None, (26, 20)),
    ('fr', ['La bibliothe\u0300que ouvre chaque jour a\u0300 neuf heures.',
            'Le calendrier de\u0301taille les horaires de chaque atelier.'], (28, 19)),
    ('ar', ['تفتح المكتبة في التاسعة كل صباح.',
            'يرجى مراجعة الجدول لمعرفة مواعيد الأنشطة خلال الأسبوع القادم.'], (28, 19)),
])
def test_shared_bubble_reflow_balances_source_sizes_without_splitting_shaped_words(
        bubble_layout, monkeypatch, reverse, language, texts, source_sizes):
    from mtu_engine.engine import configuration
    renderer, observed, native = bubble_layout
    image, mask, regions, defaults = _shared_reflow_page(source_sizes)
    texts = defaults if texts is None else texts
    if reverse:
        regions, texts = regions[::-1], texts[::-1]
    snapshot = json.loads(json.dumps(regions))
    rectangles, candidates = [], []
    rectangle, resize = native.find_largest_inscribed_rect, native.resize_regions_to_font_size

    def counted_rectangle(value):
        if value.shape != image.shape[:2]:
            rectangles.append(value.shape)
        return rectangle(value)

    def counted_resize(*args, **kwargs):
        result = resize(*args, **kwargs)
        if not args[2].render.balloon_fill_mask_layout:
            candidates.extend(args[1])
        return result

    monkeypatch.setattr(native, 'find_largest_inscribed_rect', counted_rectangle)
    monkeypatch.setattr(native, 'resize_regions_to_font_size', counted_resize)
    outputs = []
    for cache_bytes in (0, None):
        observed.clear()
        rectangles.clear()
        candidates.clear()
        output = renderer.render(image, image, regions, texts, language, mask,
                                 mask_cache_bytes=cache_bytes)
        blocks = _assert_shared_reflow_pixels(image, output, mask, regions, texts, observed, native)
        # Compare final rendered sizes with the same final-scale constraint;
        # the native integer rounding is not a raw 75% source-size promise.
        assert all(block.font_size >= native._apply_final_font_constraints(
            .75 * region['font_size'], configuration()) for block, region in zip(blocks, regions))
        assert all(block.horizontal for block in blocks)
        assert len(rectangles) <= 6  # Two ordinary ROIs + at most two two-cell plans.
        assert len(candidates) <= 22  # Six ordinary fits + four shared cells with at most four modes.
        if language == 'en':
            # Independent fitting leaves this caption below the threshold.
            # This positive reflow fixture must also exceed the raw threshold.
            assert min(block.font_size / region['font_size']
                       for block, region in zip(blocks, regions)) >= .75
            # Other scripts may keep a valid native layout without a strict
            # axis split. This reflow fixture must retain top-to-bottom order.
            top, bottom = sorted(blocks, key=lambda block: np.asarray(block.lines)[..., 1].mean())
            assert top.dst_points[..., 1].max() < bottom.dst_points[..., 1].min()
            assert len(rectangles) >= 2  # Exercise actual joint allocation, not a fast-path case.
        else:
            assert all(block.horizontal for block in candidates)
            for block, text in zip(blocks, texts):
                assert block.translation.replace('[BR]', ' ').split() == text.split()
        assert regions == snapshot
        outputs.append(output)
    np.testing.assert_array_equal(outputs[0], outputs[1])


@pytest.mark.parametrize('boundary', ['hole', 'edge', 'occupied'])
def test_shared_bubble_reflow_respects_holes_page_edges_and_external_paragraphs(
        bubble_layout, boundary):
    from manga_translator.utils import TextBlock
    renderer, observed, native = bubble_layout
    image, mask, regions, texts = _shared_reflow_page()
    if boundary == 'hole':
        mask[110:130, 90:108] = 0
    elif boundary == 'edge':
        mask[:, :-25] = mask[:, 25:]
        mask[:, -25:] = 0
        for region in regions:
            points = np.asarray(region['lines'])
            points[..., 0] -= 25
            region['lines'] = points.tolist()
    else:
        free = serialize_region(TextBlock([[[190, 190], [205, 190], [205, 205], [190, 205]]],
            ['outside the pair'], font_size=12, direction='h'))
        free['bubble_enclosed'] = False
        regions.append(free)
        texts.append('!')
    snapshot = json.loads(json.dumps(regions))
    output = renderer.render(image, image, regions, texts, 'en', mask)
    _assert_shared_reflow_pixels(image, output, mask, regions, texts, observed, native)
    assert regions == snapshot


@pytest.mark.parametrize('explicit', [False, True])
def test_shared_bubble_reflow_preserves_normal_pairs_and_explicit_breaks(
        bubble_layout, monkeypatch, explicit):
    from manga_translator.utils import TextBlock
    renderer, observed, native = bubble_layout
    image = np.full((320, 380, 3), 255, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[20:300, 20:360] = 255
    lines = [[[[40, 50], [160, 50], [160, 130], [40, 130]]],
             [[[190, 170], [330, 170], [330, 260], [190, 260]]]]
    regions = [serialize_region(TextBlock(value, ['source'], font_size=size, direction='h'))
               for value, size in zip(lines, (28, 19))]
    texts = (['This is a deliberately long explicit first line[BR]Follow the existing break.',
              'Keep this existing[BR]paragraph as written.'] if explicit else
             ['Hello there!', 'See you soon.'])
    candidates, resize = [], native.resize_regions_to_font_size

    def record(*args, **kwargs):
        result = resize(*args, **kwargs)
        if not args[2].render.balloon_fill_mask_layout:
            candidates.extend(args[1])
        return result

    monkeypatch.setattr(native, 'resize_regions_to_font_size', record)
    output = renderer.render(image, image, regions, texts, 'en', mask)
    blocks = _assert_shared_reflow_pixels(image, output, mask, regions, texts, observed, native)
    # The ordinary pass may resolve initially overlapping native boxes. Neither
    # case needs extra shared-cell fitting beyond one horizontal fit per block.
    assert len(candidates) <= len(regions)
    if explicit:
        assert min(block.font_size / region['font_size']
                   for block, region in zip(blocks, regions)) < .75
        assert [block.translation for block in blocks] == texts
    else:
        assert all(block.font_size >= .75 * region['font_size']
                   for block, region in zip(blocks, regions))


def test_shared_bubble_reflow_rolls_back_both_paragraphs_without_a_joint_gain(
        bubble_layout, monkeypatch):
    renderer, observed, native = bubble_layout
    image, mask, regions, texts = _shared_reflow_page()
    snapshot = json.loads(json.dumps(regions))
    rectangle = native.find_largest_inscribed_rect
    # Keep the native initial layouts but make every optional recovery budget
    # empty. This gives the fallback pixels without another layout implementation.
    with monkeypatch.context() as unavailable:
        unavailable.setattr(native, 'find_largest_inscribed_rect', lambda value:
            rectangle(value) if value.shape == image.shape[:2] else (0, 0, 0, 0))
        expected = renderer.render(image, image, regions, texts, 'en', mask)
    baseline = [(block.font_size, block.translation, block.dst_points.copy()) for block in observed]
    observed.clear()
    candidates, resize = [], native.resize_regions_to_font_size

    def undersized_caption(*args, **kwargs):
        result = resize(*args, **kwargs)
        if not args[2].render.balloon_fill_mask_layout:
            for block in args[1]:
                candidates.append(block.original_font_size)
                if block.original_font_size == regions[1]['font_size']:
                    block.font_size = 1  # Main-paragraph changes alone cannot commit a pair.
        return result

    monkeypatch.setattr(native, 'resize_regions_to_font_size', undersized_caption)
    actual = renderer.render(image, image, regions, texts, 'en', mask)
    assert set(candidates) == {region['font_size'] for region in regions}
    np.testing.assert_array_equal(actual, expected)
    for block, (size, text, polygon) in zip(observed, baseline):
        assert (block.font_size, block.translation) == (size, text)
        np.testing.assert_array_equal(block.dst_points, polygon)
    assert regions == snapshot
