"""Fit translated text to source boxes and bubbles with MTU's glyph measurements."""
from copy import copy
from math import hypot, sqrt
import unicodedata

import numpy as np


def _can_stack_text(text):
    """MTU stacks code points; do not split graphemes or lose bidi shaping."""
    from PyQt6.QtCore import QTextBoundaryFinder

    boundaries = QTextBoundaryFinder(QTextBoundaryFinder.BoundaryType.Grapheme, text)
    position = 0
    for char in text:
        if unicodedata.bidirectional(char) in ('R', 'AL', 'AN', 'NSM') or unicodedata.category(char) == 'Cf':
            return False
        position += 2 if ord(char) > 0xffff else 1  # Qt positions use UTF-16.
        boundaries.setPosition(position)
        if not boundaries.isAtBoundary():
            return False
    return True


def _wrap_words(text, font_size, width, letter_spacing):
    """Let Qt break at word boundaries, without Manga2Eng's short-word groups."""
    from PyQt6.QtGui import QTextLayout, QTextOption
    from manga_translator.rendering.text_render._fonts import _layout_font

    layout = QTextLayout(text, _layout_font(font_size, letter_spacing))
    option = QTextOption()
    option.setWrapMode(QTextOption.WrapMode.WordWrap)
    layout.setTextOption(option)
    encoded, lines = text.encode('utf-16-le'), []
    layout.beginLayout()
    try:
        while (line := layout.createLine()).isValid():
            line.setLineWidth(width)
            start, length = line.textStart() * 2, line.textLength() * 2
            lines.append(encoded[start:start+length].decode('utf-16-le').strip())
    finally:
        layout.endLayout()
    return '[BR]'.join(lines)


def _points_at(block, size, config):
    from manga_translator.rendering import _calc_region_dst_points_for_font
    return _calc_region_dst_points_for_font(region=block, font_size=size,
        render_horizontally=block.horizontal, line_spacing_multiplier=block.line_spacing,
        letter_spacing_multiplier=block.letter_spacing, config=config, anchor_mode='center')


def select_region(block, config):
    from manga_translator.rendering import text_render
    config._current_region = block
    text_render.set_font(block.font_family)


def resize_unscaled(image, blocks, config, original=None, mask=None):
    from manga_translator.rendering import resize_regions_to_font_size
    scale = config.render.font_scale_ratio
    config.render.font_scale_ratio = 1.0
    try:
        return resize_regions_to_font_size(image, blocks, config, original, bubble_mask=mask,
                                           skip_text_replacements=True)
    finally:
        config.render.font_scale_ratio = scale


def _reserve_source_size(block, polygon, config, *, unscaled=False):
    from manga_translator import rendering as native

    base = min(block.original_font_size, block.font_size) if unscaled else block.original_font_size
    maximum = native._apply_final_font_constraints(base, config)
    if block.font_size <= maximum:
        return polygon
    block.font_size = maximum
    select_region(block, config)
    resized = _points_at(block, maximum, config)
    return resized if resized is not None else polygon


def grow_into_space(block, text, polygon, width, height, config, fits, center=None):
    """Try one reflow; text=None keeps the breaks of a recovered layout."""
    from manga_translator import rendering as native

    if polygon is None or min(width, height) <= 0:
        return block, polygon
    corners = polygon.reshape(4, 2)
    ink_w, ink_h = hypot(*(corners[1] - corners[0])), hypot(*(corners[2] - corners[1]))
    if not (ink_w > 0 and ink_h > 0):
        return block, polygon
    fixed = text is None or block.translation_rich or native._has_explicit_line_breaks(text)
    # Reflow can use the whole area; fixed breaks must fit in both dimensions.
    scale = min(width / ink_w, height / ink_h) if fixed else sqrt(width * height / (ink_w * ink_h))
    capacity = native._apply_final_font_constraints(min(block.font_size * scale, max(width, height), 8192), config)
    source = block.original_font_size
    # Use a quarter of the spare size for legibility, at most one small type step.
    extra = min(4, source / 4, max(0, capacity - source) / 4)
    target = min(capacity, source + int(extra))
    if target <= block.font_size:
        return block, polygon

    candidate = copy(block)
    select_region(candidate, config)
    if not fixed:
        # Balance the aspect ratio only when the paragraph needs shrinking.
        span = width if block.horizontal else height
        if target < source:
            span *= target / (block.font_size * scale)
        if block.horizontal:
            candidate.translation = _wrap_words(text, target, max(1, span), block.letter_spacing)
        elif _can_stack_text(text):
            from manga_translator.rendering.auto_linebreak import _layout_vertical
            lines, _ = _layout_vertical(target, text, max(1, span), config, block.letter_spacing)
            candidate.translation = '[BR]'.join(lines)

    # Try the target first, then bisect only sizes better than the drawable baseline.
    lo, hi, size = int(block.font_size) + 1, target, target
    best = None
    while lo <= hi:
        points = _points_at(candidate, size, config)
        at = block.center
        if points is not None and center is not None and fits(centered := points + (center - at)):
            points, at = centered, center
        elif points is not None and not fits(points):
            points = None
        if points is not None:
            best = size, at, points
            lo = size + 1
        else:
            hi = size - 1
        size = (lo + hi) // 2
    if best is None:
        return block, polygon
    candidate.font_size, candidate.center, points = best
    return candidate, points


def _inside_source_box(polygon, source):
    import cv2
    return all(cv2.pointPolygonTest(source, tuple(map(float, pt)), False) >= 0 for pt in polygon.reshape(-1, 2))


def _grow_in_source_box(block, text, polygon, shape, config, obstacles):
    from manga_translator.rendering import _layout_regions_conflict
    height, width = shape[:2]
    source = block.min_rect[0].astype(np.float32)

    def fits(p):
        corners = p.reshape(-1, 2)
        return ((corners >= (2, 2)).all() and (corners <= (width-2, height-2)).all()
            and _inside_source_box(p, source)
            and not _layout_regions_conflict(p, None, obstacles))

    return grow_into_space(block, text, polygon, *block.unrotated_size, config, fits)


def _bounds(polygon):
    vertices = np.asarray(polygon).reshape(-1, 2)
    return np.concatenate((vertices.min(axis=0), vertices.max(axis=0)))


def _neighbours(polygons, boxes, index, bounds):
    left, top, right, bottom = bounds
    nearby = (boxes[:, 0] < right) & (boxes[:, 2] > left) & (boxes[:, 1] < bottom) & (boxes[:, 3] > top)
    nearby[index] = False
    return [(polygons[i], None) for i in np.flatnonzero(nearby)]


def fit_source_boxes(image, blocks, texts, config, occupied):
    from manga_translator import rendering as native

    points = native.resize_regions_to_font_size(image, blocks, config, None, skip_text_replacements=True)
    points = [_reserve_source_size(b, p, config) for b, p in zip(blocks, points)]
    reserved = [b.min_rect if p is None else p for b, p in zip(blocks, points)] + occupied
    boxes = np.array([_bounds(p) for p in reserved])
    height, width = image.shape[:2]
    for i, (block, text, polygon) in enumerate(zip(blocks, texts, points)):
        if polygon is None:
            continue
        obstacles = _neighbours(reserved, boxes, i, _bounds(block.min_rect))
        block, polygon = _grow_in_source_box(block, text, polygon, image.shape, config, obstacles)
        bounds = _bounds(polygon)
        left, top, right, bottom = bounds
        if not (left >= 2 and top >= 2 and right <= width - 2 and bottom <= height - 2):
            # Scale the rotated glyph box to the page margins, then measure it once at the final size.
            cx, cy = block.center
            extent = np.maximum([cx - left, right - cx, cy - top, bottom - cy], 1)
            budget = np.maximum([cx - 2, width - 2 - cx, cy - 2, height - 2 - cy], 1)
            ratio = min(1.0, float(np.min(budget / extent)))
            select_region(block, config)
            options = dict(text=block.get_translation_for_rendering(), is_horizontal=block.horizontal,
                line_spacing=block.line_spacing, config=config, target_lang=block.target_lang,
                letter_spacing=block.letter_spacing, stroke_width=block.stroke_width)
            w, h, _, _ = native.calc_box_from_font(block.font_size, **options)
            block.font_size = min(block.font_size, native.calc_font_from_box(w * ratio, h * ratio, **options))
            resized, _ = native.calc_box_from_font(block.font_size, center=block.center, angle=block.angle, **options)
            polygon = resized if resized is not None else block.min_rect
            bounds = _bounds(polygon)
        blocks[i], boxes[i] = block, bounds
        points[i] = reserved[i] = polygon
    return points


def associate_bubbles(blocks, mask):
    has_mask = mask is not None and np.any(mask)
    if has_mask:
        from manga_translator.rendering import _region_lines_fully_inside_mask
    for block in blocks:
        if getattr(block, 'bubble_enclosed', None) is None:
            block.bubble_enclosed = bool(has_mask and
                _region_lines_fully_inside_mask(block, mask))


def fit_bubbles(image, original, blocks, texts, config, mask, labels, occupied=(), *, language, unscaled_points=None):
    """Reserve all paragraphs before recovery and growth; never move their source lines."""
    import cv2
    from manga_translator import rendering as native
    from manga_translator.utils import build_region_reference_mask
    from manga_translator.utils.bubble import reference_mask_bounds

    points = unscaled_points
    if points is None:
        points, pending = [None] * len(blocks), []
        for i, block in enumerate(blocks):
            # A fitting native single line needs no balloon search for its baseline.
            if (not config.render.balloon_fill_mask_layout and native.should_force_no_wrap_single_region(block)
                    and not native._has_explicit_line_breaks(texts[i])):
                size = native._apply_final_font_constraints(block.original_font_size, config)
                select_region(block, config)
                polygon = _points_at(block, size, config)
                source = block.min_rect[0].astype(np.float32)
                if polygon is not None and _inside_source_box(polygon, source):
                    block.font_size, points[i] = size, polygon
                    continue
            pending.append(i)
        if pending:
            fitted = native.resize_regions_to_font_size(image, [blocks[i] for i in pending], config, original,
                bubble_mask=mask, skip_text_replacements=True)
            for i, polygon in zip(pending, fitted):
                points[i] = polygon
    points = [_reserve_source_size(block, polygon, config, unscaled=unscaled_points is not None)
              for block, polygon in zip(blocks, points)]
    if mask is None or labels is None:
        return blocks, points
    height, width = image.shape[:2]
    reserved = [*points, *occupied]
    boxes = np.array([_bounds(p) for p in reserved])

    def safe(polygon, reference):
        if polygon is None:
            return False
        vertices = np.asarray(polygon).reshape(-1, 2)
        # MTU clips polygons to the mask, so check the physical page first.
        return bool(np.isfinite(vertices).all() and
            (vertices >= (1, 1)).all() and (vertices <= (width - 1, height - 1)).all() and
            native._polygon_fully_inside_mask(vertices, reference))

    def avoid(block, polygon, reference, obstacles):
        if polygon is None:
            return None
        select_region(block, config)
        options = dict(region=block, min_font_size=1, render_horizontally=block.horizontal,
            line_spacing_multiplier=block.line_spacing, letter_spacing_multiplier=block.letter_spacing,
            config=config, anchor_mode='center', bubble_mask=reference)
        if not safe(polygon, reference):
            size, polygon = native._binary_search_font_for_bubble_mask(
                start_font_size=block.font_size, **options)
            if size is None:
                return None
            block.font_size = size
        if native._layout_regions_conflict(polygon, reference, obstacles):
            size, polygon = native._shrink_font_for_layout_collisions(
                start_font_size=block.font_size, placed_regions=obstacles, **options)
            if size is not None:
                block.font_size = size
        return polygon if safe(polygon, reference) else None

    def available_space(reference, bounds, obstacles):
        x, y, w, h = bounds
        available = reference[y:y+h, x:x+w].copy()
        if w and h:
            if x == 0:
                available[:, 0] = 0
            if y == 0:
                available[0, :] = 0
            if x + w == width:
                available[:, -1] = 0
            if y + h == height:
                available[-1, :] = 0
            for polygon, _ in obstacles:
                cv2.fillPoly(available, [np.rint(np.asarray(polygon).reshape(-1, 2) - (x, y)).astype(np.int32)], 0)
        return available

    def reset_text(block, text):
        candidate = copy(block)
        candidate.translation = text
        candidate.translation_rich = block.translation_rich
        candidate.font_size = block.original_font_size
        return candidate

    def fit_rectangle(block, text, rect, reference, obstacles, *, recover, best, cell=None):
        chosen, chosen_points = best
        rx, ry, rw, rh = rect
        if rw <= 2 or rh <= 2:
            return best
        vertical = recover and _can_stack_text(text)
        # Stacking and native hyphenation both split code points.
        hyphenate = (vertical and not block.translation_rich and
                     native.text_render.select_hyphenator(language) is not None)
        modes = ('horizontal', 'vertical', 'columns') if vertical else ('horizontal',)
        if cell is not None and hyphenate:
            modes = ('words', *modes)
        for mode in modes:
            stack = mode in ('vertical', 'columns')
            if stack and chosen_points is not None and chosen.font_size >= .75 * block.original_font_size:
                break
            candidate = reset_text(block, text)
            candidate.center = np.array([rx + rw / 2, ry + rh / 2])
            candidate.unrotated_size = (rw - 2, rh - 2)
            # Preserve a healthy shared paragraph's wrap size.
            seed = round(block.font_size if cell is not None and
                block.font_size >= .75 * block.original_font_size else block.original_font_size)
            select_region(candidate, config)
            mask_layout = config.render.balloon_fill_mask_layout
            english_layout = config.render.bubble_layout_english
            config.render.balloon_fill_mask_layout = False
            if stack:
                candidate._direction = 'v'
                config.render.bubble_layout_english = False
            try:
                if mode == 'columns':
                    from manga_translator.rendering.auto_linebreak import solve_no_br_layout
                    # The rectangle is a layout budget, not a single OCR column.
                    config._current_region = None
                    try:
                        wrapped = solve_no_br_layout(text=text, horizontal=False,
                            seed_segments=0, seed_font_size=round(block.original_font_size),
                            bubble_width=rw-2, bubble_height=rh-2, min_font_size=1,
                            max_font_size=round(block.original_font_size),
                            line_spacing_multiplier=candidate.line_spacing,
                            letter_spacing_multiplier=candidate.letter_spacing,
                            target_lang=candidate.target_lang, config=config)
                        candidate.translation = wrapped.text_with_br
                    finally:
                        config._current_region = candidate
                    if candidate.translation == text:
                        continue
                elif mode == 'horizontal' and hyphenate:
                    from manga_translator.rendering.auto_linebreak import _layout_horizontal_eng
                    lines, _ = _layout_horizontal_eng(seed, text, rw - 2,
                        language=language, hyphenate=True, letter_spacing=candidate.letter_spacing)
                    candidate.translation = '[BR]'.join(lines)
                elif not stack and recover and not block.translation_rich:
                    candidate.translation = _wrap_words(text, seed,
                        rw - 2, candidate.letter_spacing)
                else:
                    native._apply_default_english_line_break_method(candidate, round(candidate.font_size), None, config)
                polygon = native.resize_regions_to_font_size(image, [candidate], config, original,
                    bubble_mask=mask, skip_text_replacements=True)[0]
            finally:
                config.render.balloon_fill_mask_layout = mask_layout
                config.render.bubble_layout_english = english_layout
            polygon = avoid(candidate, _reserve_source_size(candidate, polygon, config), reference, obstacles)
            if polygon is not None and cell is not None:
                origin, budget = cell
                vertices = np.asarray(polygon).reshape(-1, 2) - origin
                if not ((vertices >= 0).all() and
                        (vertices <= (budget.shape[1]-1, budget.shape[0]-1)).all() and
                        native._polygon_fully_inside_mask(vertices, budget)):
                    polygon = None
            if polygon is not None and (chosen_points is None or candidate.font_size > chosen.font_size):
                chosen, chosen_points = candidate, polygon
        return chosen, chosen_points

    def neighbours(index, bounds):
        x, y, w, h = bounds
        return _neighbours(reserved, boxes, index, (x, y, x+w, y+h))

    def place(index, block, polygon):
        blocks[index] = block
        points[index] = reserved[index] = polygon
        boxes[index] = _bounds(polygon)

    def grow(index, text, bounds, reference, obstacles):
        block = blocks[index]
        x, y, w, h = bounds
        room_w, room_h = w - 1, h - 1
        center = np.array([x + room_w / 2, y + room_h / 2])
        if obstacles:
            cx, cy = block.center
            room_w, room_h = 2 * min(cx - x, x + w - 1 - cx), 2 * min(cy - y, y + h - 1 - cy)
            center = None
        elif max(abs(center - block.center)) < 1:
            center = None

        def fits(p):
            nonlocal reference
            if reference is None:
                reference = build_region_reference_mask(block, mask, labels)
            return safe(p, reference) and not native._layout_regions_conflict(p, reference, obstacles)

        chosen, polygon = grow_into_space(block, text, points[index], room_w, room_h, config, fits, center)
        if chosen is not block:
            place(index, chosen, polygon)

    shared = []
    for index, block in enumerate(blocks):
        reference = build_region_reference_mask(block, mask, labels)
        bounds = reference_mask_bounds(reference) or cv2.boundingRect(reference)
        x, y, w, h = bounds
        obstacles = neighbours(index, bounds)
        collision = native._layout_regions_conflict(points[index], reference, obstacles)
        # Try the unused height of a narrow paragraph before accepting severe shrink.
        source_width, source_height = block.unrotated_size
        recover = (config.render.balloon_fill_mask_layout and block.horizontal and
            (h > w or source_height > source_width) and
            block.font_size < .75 * block.original_font_size and
            2 * (boxes[index, 3] - boxes[index, 1]) < h and
            not native._has_explicit_line_breaks(texts[index]))
        if (recover or collision or not safe(points[index], reference) or
                not native._polygon_fully_inside_mask(block.min_rect[0], reference)):
            chosen = copy(block)
            chosen_points = avoid(chosen, points[index], reference, obstacles)

            if w and h:
                rx, ry, rw, rh = native.find_largest_inscribed_rect(available_space(reference, bounds, obstacles))
                chosen, chosen_points = fit_rectangle(block, texts[index], (x+rx, y+ry, rw, rh),
                    reference, obstacles, recover=recover, best=(chosen, chosen_points))
            if chosen_points is None:
                # A bad mask is only a hint: paint the translation in its source box.
                chosen = reset_text(block, texts[index])
                mode = config.render.layout_mode
                config.render.layout_mode = 'smart_scaling'
                try:
                    chosen_points = native.resize_regions_to_font_size(image, [chosen], config, None,
                        skip_text_replacements=True)[0]
                finally:
                    config.render.layout_mode = mode
                chosen_points = _reserve_source_size(chosen, chosen_points, config)
                obstacles = neighbours(index, cv2.boundingRect(chosen.min_rect[0].astype(np.float32)))
                chosen, chosen_points = _grow_in_source_box(chosen, texts[index], chosen_points, image.shape,
                                                           config, obstacles)
                place(index, chosen, chosen_points)
                continue
            place(index, chosen, chosen_points)
        text = None if recover else texts[index]
        if obstacles:
            # Let every neighbour finish recovery before spending any shared space.
            shared.append((index, text, bounds))
        else:
            grow(index, text, bounds, reference, obstacles)

    # An undersized pair can share space more fairly when fitted together in source order.
    if config.render.balloon_fill_mask_layout and any(
            block.font_size < .75 * block.original_font_size for block in blocks):
        from manga_translator.utils.bubble import _region_labels
        groups = {}
        for index, block in enumerate(blocks):
            key = _region_labels(block, mask, labels)
            if len(key) == 1:
                groups.setdefault(key, []).append(index)
        for indices in groups.values():
            if len(indices) != 2 or any(not blocks[i].horizontal or blocks[i].translation_rich or
                    native._has_explicit_line_breaks(texts[i]) for i in indices):
                continue
            score = min(blocks[i].font_size / blocks[i].original_font_size for i in indices)
            if score >= .75:
                continue
            reference = build_region_reference_mask(blocks[indices[0]], mask, labels)
            obstacles = [(p, None) for i, p in enumerate(reserved) if i not in indices]
            bounds = reference_mask_bounds(reference) or cv2.boundingRect(reference)
            x, y = bounds[:2]
            available = available_space(reference, bounds, obstacles)
            origin = np.array([x, y])
            centers = {i: _bounds(blocks[i].lines).reshape(2, 2).mean(axis=0) for i in indices}

            def fit_pair(axis, split):
                order = sorted(indices, key=lambda i: centers[i][axis])
                if not 0 < split < available.shape[1-axis]:
                    return None
                pair = []
                for side, i in enumerate(order):
                    budget = available.copy()
                    excluded = [slice(None), slice(None)]
                    excluded[1-axis] = slice(split, None) if side == 0 else slice(None, split)
                    budget[tuple(excluded)] = 0
                    # Treat the divider as an obstacle so an outline can shrink to fit.
                    low, high = origin.copy(), origin + available.shape[::-1]
                    if side == 0:
                        low[axis] += split - 1
                    else:
                        high[axis] = origin[axis] + split
                    wall = np.array([low, (high[0], low[1]), high, (low[0], high[1])])
                    rx, ry, rw, rh = native.find_largest_inscribed_rect(budget)
                    candidate, polygon = fit_rectangle(blocks[i], texts[i], (x+rx, y+ry, rw, rh),
                        reference, [*obstacles, (wall, None)], recover=True,
                        best=(copy(blocks[i]), None), cell=(origin, budget))
                    if polygon is None:
                        return None
                    pair.append((i, candidate, polygon))
                if native._polygons_overlap(pair[0][2], pair[1][2]):
                    return None
                return pair

            best_pair = None
            for axis in (0, 1):
                if centers[indices[0]][axis] == centers[indices[1]][axis]:
                    continue
                split = round(sum(centers[i][axis] for i in indices) / 2 - origin[axis])
                pair = fit_pair(axis, split)
                if pair is None:
                    continue
                candidate_score = min(b.font_size / b.original_font_size for _, b, _ in pair)
                # Compare sizes after the same final scale and integer rounding.
                if (candidate_score > score and all(b.font_size >= min(blocks[i].font_size,
                        native._apply_final_font_constraints(.75 * b.original_font_size, config))
                        for i, b, _ in pair)):
                    score, best_pair = candidate_score, pair
            if best_pair is not None:
                for i, block, polygon in best_pair:
                    place(i, block, polygon)
    for index, text, bounds in shared:
        grow(index, text, bounds, None, neighbours(index, bounds))
    return blocks, points
