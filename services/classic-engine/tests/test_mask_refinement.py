"""Source-mask and bubble protection without model assets."""
from textwrap import dedent

import cv2
import numpy as np
import pytest

from mtu_engine.assets import LOCK
from mtu_engine.masks import merge_components


def test_component_votes_match_strict_xor_improvement_across_bands():
    rng = np.random.default_rng(71)
    labels = rng.integers(0, 13, (530, 512), dtype=np.uint16)
    pred = rng.integers(0, 256, labels.shape, dtype=np.uint8)
    merged = (rng.random(labels.shape) > .7).astype(np.uint8) * 255
    # Two fresh pixels on opposite sides of the band boundary make an exact tie.
    labels[0, 0] = labels[-1, 0] = 13
    merged[0, 0] = merged[-1, 0] = 0
    pred[0, 0], pred[-1, 0] = 0, 255
    expected = merged.copy()
    eligible = np.arange(14) % 3 != 0
    for label in np.flatnonzero(eligible):
        candidate = expected.copy()
        candidate[labels == label] = 255
        if (candidate ^ pred).sum() < (expected ^ pred).sum():
            expected = candidate
    merge_components(merged, pred, labels, eligible)
    np.testing.assert_array_equal(merged, expected)


@pytest.mark.parametrize('dtype', [np.uint8, np.int16, np.float32])
def test_bubble_clipping_keeps_holes_negative_values_and_owned_output(dtype):
    patches = LOCK['source']['patches']['manga_translator/mask_refinement/__init__.py']
    patch = next(p for p in patches if p['before'].startswith('def _clip_refined_components'))
    versions = []
    for key in ('before', 'after'):
        scope = dict(np=np, cv2=cv2, Tuple=tuple)
        exec(patch[key], scope)
        versions.append(scope['_clip_refined_components_by_bubble_mask'])
    rng = np.random.default_rng(904)
    for _ in range(20):
        mask = (rng.random((43, 61)) > .7).astype(dtype) * 255
        bubble = (rng.random(mask.shape) > .4).astype(dtype) * 255
        if dtype != np.uint8:
            mask[mask == 0] = -1
            bubble[bubble == 0] = -1
        snapshots = mask.copy(), bubble.copy()
        expected, actual = (fn(mask, bubble) for fn in versions)
        assert expected[1:] == actual[1:]
        np.testing.assert_array_equal(expected[0], actual[0])
        assert actual[0].dtype == np.uint8 and not np.shares_memory(actual[0], mask)
        np.testing.assert_array_equal(mask, snapshots[0])
        np.testing.assert_array_equal(bubble, snapshots[1])


def protected_refinement(raw, refined, lines, bubble=None):
    """Execute the shipped patch snippets without importing model dependencies."""
    patches = LOCK['source']['patches']['manga_translator/mask_refinement/__init__.py']
    support = next(p['after'] for p in patches if p['before'].startswith('    if use_model_bubble'))
    scope = dict(np=np, cv2=cv2, raw_mask=raw, final_mask=refined.copy(), text_regions=[],
                 kernel_size=3, _build_line_protect_mask=lambda regions, shape, expand_px: lines)
    exec(dedent(support[:support.rindex('    if use_model_bubble')]), scope)
    if bubble is None:
        return scope['final_mask']
    clip = next(p['after'] for p in patches if p['before'].startswith('def _clip_refined_components'))
    scope['Tuple'] = tuple
    exec(clip, scope)
    scope['mask_before_clip'] = scope['final_mask']
    scope['clipped_mask'] = scope['_clip_refined_components_by_bubble_mask'](scope['final_mask'], bubble)[0]
    restore = next(p['after'] for p in patches if 'protected_restore_mask =' in p['before'])
    exec(dedent(restore), scope)
    scope['clipped_mask'][scope['protected_restore_mask']] = 255
    return scope['clipped_mask']


@pytest.mark.parametrize('resized', [False, True])
def test_refinement_restores_detected_strokes_not_entire_selected_line_boxes(resized):
    raw = np.zeros((40, 60), np.uint8)
    raw[14:17, 16:26] = 255
    raw[30:33, 45:50] = 255  # An unselected paragraph must not be erased.
    raw[3, 3] = 80  # Weak support outside selected text must not be added either.
    raw = cv2.resize(raw, (120, 80), interpolation=cv2.INTER_NEAREST) if resized else raw
    raw.flags.writeable = False
    lines = np.zeros((40, 60), np.uint8)
    lines[9:23, 10:32] = 255
    refined = np.zeros_like(lines)
    refined[12:20, 12:30] = 255
    refined[15, :] = 0  # Refinement left a seam through known text.
    snapshots = raw.copy(), refined.copy(), lines.copy()
    result = protected_refinement(raw, refined, lines)
    assert np.all(result[14:17, 16:26] == 255)
    assert result[15, 14] == 255  # Native 2px guard closes the seam beside the stroke.
    assert result[9, 10] == 0 and result[31, 47] == 0 and result[3, 3] == 0
    assert not np.any((result > 0) & (refined == 0) & (lines == 0))
    for actual, original in zip((raw, refined, lines), snapshots):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize('edge', range(4))
def test_bubble_clip_preserves_source_at_page_edges_but_keeps_holes_and_unselected_text(edge):
    raw = np.zeros((50, 50), np.uint8)
    raw[1:7, 12:33] = 255
    raw[1:3, 21:24] = 0  # A short gap in the raw mask must not seed LaMa ghosts.
    raw[38:42, 12:33] = 255  # A separate unselected paragraph.
    refined = np.zeros_like(raw)
    refined[:30, 10:36] = 255
    lines = np.zeros_like(raw)
    lines[4:10, 12:33] = 255
    bubble = np.zeros_like(raw)
    bubble[3:36, 8:38] = 255
    bubble[19:23, 19:25] = 0
    raw, refined, lines, bubble = (np.rot90(a, edge).copy() for a in (raw, refined, lines, bubble))
    result = np.rot90(protected_refinement(raw, refined, lines, bubble), -edge)
    assert np.all(result[:3, 12:33] == 255)
    assert result[20, 21] == 0  # A genuine bubble hole, without text support.
    assert result[0, 35] == 0  # Unsupported dilation is still clipped.
    assert not np.any(result[38:42])
