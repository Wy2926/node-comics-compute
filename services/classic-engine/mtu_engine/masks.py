"""Adapt pinned CTD refinement to selected lines and MTU's inpainting margin."""
from types import SimpleNamespace

import cv2
import numpy as np

from .backend import check_cancelled


def merge_components(merged, pred, labels, eligible):
    """Score disjoint CTD components together against the current merged mask."""
    if not eligible.any():
        return
    gain = np.zeros(len(eligible), dtype=np.float64)
    rows = max(1, 256 * 1024 // labels.shape[1])
    for y in range(0, len(labels), rows):
        band = slice(y, y + rows)
        fresh = merged[band] == 0
        # At a newly set pixel the XOR error falls from p to 255-p. Ties stay unset.
        weights = 2 * pred[band][fresh].astype(np.int16) - 255
        gain += np.bincount(labels[band][fresh], weights=weights, minlength=len(eligible))
    keep = eligible & (gain > 0)
    for y in range(0, len(labels), rows):
        band = slice(y, y + rows)
        merged[band][keep[labels[band]]] = 255


def complete_mask(rgb, mask, textlines, dilation_offset=0, kernel_size=3):
    from manga_translator.detection.ctd_utils.textmask import REFINEMASK_ANNOTATION, refine_mask
    from manga_translator.detection.ctd_utils.utils.imgproc_utils import enlarge_window
    from manga_translator.utils import Quadrilateral
    from manga_translator.utils.bubble import _region_labels

    check_cancelled()
    final = np.zeros_like(mask)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    for line in textlines:
        check_cancelled()
        ids = np.array(_region_labels(SimpleNamespace(lines=[line.pts]), mask, labels), dtype=np.int32)
        ids = ids[stats[ids, cv2.CC_STAT_AREA] > 9]
        if not ids.size:
            continue
        # Keep complete source components: a detected polygon can cut through a glyph.
        boxes = stats[ids, :4]
        x, y = boxes[:, :2].min(axis=0)
        r, b = (boxes[:, :2] + boxes[:, 2:]).max(axis=0)
        component = Quadrilateral(np.array([[x, y], [r, y], [r, b], [x, b]]), '', 0)
        size = max((int((line.font_size + dilation_offset) * .3) // 2) * 2 + 1, 3)
        x1, y1, x2, y2 = enlarge_window(component.xyxy, rgb.shape[1], rgb.shape[0])
        x1, y1 = max(0, x1 - size // 2), max(0, y1 - size // 2)
        x2, y2 = min(rgb.shape[1], x2 + size // 2), min(rgb.shape[0], y2 + size // 2)
        local = Quadrilateral(component.pts - (x1, y1), '', 0)
        pred = np.isin(labels[y1:y2, x1:x2], ids).astype(np.uint8) * 255
        # MTU's CTD expects BGR; annotation mode leaves dilation to MTU's margin.
        refined = refine_mask(rgb[y1:y2, x1:x2, ::-1], pred, [local], refine_mode=REFINEMASK_ANNOTATION)
        refined |= pred  # Threshold selection must not remove known source strokes.
        refined = cv2.dilate(refined, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
        final[y1:y2, x1:x2] |= refined
    return cv2.dilate(final, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
