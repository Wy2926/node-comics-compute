"""Assemble the node's MTU image and lettering stages."""
import asyncio
from contextlib import nullcontext
import json
import logging
import os
from pathlib import Path
from threading import Lock, current_thread, main_thread
from time import perf_counter

import numpy as np

from classic_node.timing import record, stage, waiting_for

from .assets import activate
from .backend import check_cancelled, device_context, validate_backend
from .colors import ensure_stroke_contrast, sample_colors
from .fonts import FONT_CHOICES, font_filename
from .layout import associate_bubbles, fit_bubbles, fit_source_boxes, resize_unscaled, select_region
from .layout_metrics import page_measurements

LANGUAGES = {'zh-Hans': 'CHS', 'zh-Hant': 'CHT', 'ja': 'JPN', 'ko': 'KOR', 'en': 'ENG',
             'fr': 'FRA', 'es': 'ESP', 'pt-BR': 'PTB', 'de': 'DEU', 'it': 'ITA', 'ru': 'RUS',
             'pl': 'POL', 'uk': 'UKR', 'tr': 'TRK', 'vi': 'VIN', 'id': 'IND', 'ar': 'ARA'}


def configuration(detect_size=1280, inpainting_size=512):
    from manga_translator.config import Config
    return Config.model_validate({
        'detector': {'detector': 'default', 'detection_size': detect_size,
                     'text_threshold': 0.5, 'box_threshold': 0.6, 'unclip_ratio': 2.3},
        'ocr': {'limit_mask_dilation_to_bubble_mask': True},
        'inpainter': {'inpainter': 'lama_large', 'inpainting_size': inpainting_size,
                      'inpainting_precision': 'fp32', 'force_use_torch_inpainting': True,
                      'solid_fill_pure_bubbles': True, 'per_block_inpainting': True},
        'render': {'renderer': 'default', 'layout_mode': 'smart_scaling',
                   'balloon_fill_mask_layout': False, 'center_text_in_bubble': True,
                   'no_hyphenation': True,
                   'stroke_width': 0.1, 'font_scale_ratio': 0.96},
        'translator': {'translator': 'none'},
    })


def serialize_region(region):
    # Preserve raw geometry. MTU's editor export intentionally unrotates lines,
    # so the compute checkpoint stores the constructor fields directly.
    return {'lines': region.lines.tolist(), 'texts': list(region.texts),
            'font_size': float(region.font_size), 'angle': float(region.angle),
            'fg_color': list(map(int, region.fg_colors)), 'bg_color': list(map(int, region.bg_colors)),
            'prob': float(region.prob), 'direction': region._direction,
            'default_stroke_width': float(region.default_stroke_width),
            'adjust_bg_color': bool(region.adjust_bg_color),
            'line_spacing': float(region.line_spacing), 'letter_spacing': float(region.letter_spacing),
            'bubble_enclosed': getattr(region, 'bubble_enclosed', None)}


class Renderer:
    def __init__(self, models, fonts):
        if current_thread() is not main_thread():
            raise RuntimeError('Initialize Qt on the node main thread before starting workers')
        root = Path(models).resolve().parent
        activate(models)
        os.environ['BALLOONTRANS_APP_ROOT'] = str(root / 'upstream')
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        os.environ['QT_API'] = 'pyqt6'
        os.environ['QT_QPA_FONTDIR'] = str(root / 'fonts')
        os.environ['TQDM_DISABLE'] = '1'
        # Upstream debug/info may contain OCR or translation text. Only our
        # bounded, sanitized node diagnostics enter the operational log.
        logger = logging.getLogger('manga-translator')
        logger.handlers = [logging.NullHandler()]
        logger.propagate = False
        logger.setLevel(logging.CRITICAL + 1)
        from hyphen import dictools
        dictools.DEFAULT_DICT_PATH = str(root / 'hyphenation')
        registry = json.loads((root / 'hyphenation/dictionaries.json').read_text(encoding='utf-8'))
        from manga_translator.rendering import text_render
        # Upstream rewrites aliases at import. The prepared offline registry is
        # authoritative; install it after that import, before any dictionary use.
        dictools.LANGUAGES[:] = list(registry)
        from PyQt6.QtGui import QRawFont
        self.fonts = tuple(fonts)
        self.font_by_name = {Path(path).name: path for path in self.fonts}
        if not self.fonts:
            raise ValueError('Use the prepared bundle fonts')
        text_render.set_font(self.fonts[0])
        self.families = [text_render.register_font_file(path)[0] for path in self.fonts]
        # Register and initialize before render workers start. Qt owns fallback,
        # shaping, line breaking, size search, glyph rasterization and compositing.
        text_render.set_font(self.families[0])
        self.raw_fonts = [QRawFont(path, 24) for path in self.fonts]
        if not all(font.isValid() for font in self.raw_fonts):
            raise ValueError('Invalid prepared font')
        self.font_ids = [key for key, (filename, _) in FONT_CHOICES.items() if filename in self.font_by_name]
        self._lock = Lock()

    def covers(self, text):
        return all(any(font.supportsCharacter(ord(char)) for font in self.raw_fonts)
                   for char in text if not char.isspace())

    def render(self, original, cleaned, regions, translations, language, bubble_mask, *, mask_cache_bytes=None,
               bubble_prepared=False, bubble_components=None, font_id=None):
        from manga_translator.config import Direction
        from manga_translator.utils import TextBlock, erode_bubble_mask
        from manga_translator.rendering.rich_text import legacy_line_breaks_to_document
        config = configuration()
        target = LANGUAGES[language]
        config.translator.target_lang = target
        font = font_filename(language, font_id)
        if font not in self.font_by_name:
            raise ValueError('Translation font is not installed')
        config.render.font_family = self.font_by_name[font]
        word_layout = language not in ('zh-Hans', 'zh-Hant', 'ja')
        if word_layout:
            config.render.direction = Direction.h
            # MTU explicitly supports this word-based layout switch for every
            # language. Preserve words (and Arabic joins) before size search.
            config.render.bubble_layout_english = language != 'en'
        blocks = []
        for data, text in zip(regions, translations):
            if not text.strip():
                continue
            block = TextBlock(**data)
            block.bubble_enclosed = data.get('bubble_enclosed')
            ensure_stroke_contrast(block)
            block.translation = text
            block.target_lang = target
            block.font_family = config.render.font_family
            block.default_stroke_width = config.render.stroke_width
            if word_layout:
                block._direction = config.render.direction.value
                if len(text.split()) == 1:
                    # MTU's plain-text fallback can split even a single word.
                    block.set_translation_rich(legacy_line_breaks_to_document(text))
            blocks.append(block)
        if not blocks:
            return cleaned
        if mask_cache_bytes is None:
            mask_cache_bytes = cleaned.shape[0] * cleaned.shape[1] * len(blocks)
        # MTU's font selection is thread-local; its registry/hyphenator caches
        # are shared. Bound the whole upstream call until concurrency is proven.
        with self._lock, page_measurements():
            if bubble_mask is not None and not bubble_prepared:
                with stage('render_bubble_prepare'):
                    associate_bubbles(blocks, bubble_mask)
                    bubble_mask = erode_bubble_mask(bubble_mask, 0.02)
            return self._render_blocks(original, cleaned, blocks, language, config, bubble_mask,
                                       bubble_components, mask_cache_bytes, word_layout)

    def _render_blocks(self, original, cleaned, blocks, language, config, bubble_mask, components, cache_bytes,
                       word_layout):
        from manga_translator.rendering import (dispatch, _region_lines_fully_inside_mask,
            _apply_default_english_line_break_method, _apply_default_english_case_preferences,
            _has_explicit_line_breaks, _resolve_initial_layout_font_size)
        from manga_translator.utils import build_region_reference_mask
        from manga_translator.utils.bubble import reference_mask_cache, cached_bubble_labels
        with reference_mask_cache(cache_bytes, bubble_mask=bubble_mask, components=components, record=record):
            enclosed, free = [], []
            labels = None
            if bubble_mask is not None and np.any(bubble_mask):
                # Reuse the same native 2% inset and labels as solid-fill.
                labels = cached_bubble_labels(bubble_mask)
            for block in blocks:
                # Also retain the source scale for break-only paragraphs that MTU skips measuring.
                block.original_font_size = _resolve_initial_layout_font_size(block, cleaned, config)
                # Ownership uses the original mask, not its inset for padding.
                # Legacy/direct prepared inputs without metadata stay strict.
                inside = block.bubble_enclosed
                if inside is None and labels is not None:
                    inside = _region_lines_fully_inside_mask(block,
                        build_region_reference_mask(block, bubble_mask, labels))
                (enclosed if inside and labels is not None else free).append(block)
            enclosed_texts = [block.translation for block in enclosed]
            free_texts = [block.translation for block in free]
            rewrap = word_layout and any(not b.translation_rich and not _has_explicit_line_breaks(b.translation)
                                         for b in enclosed)
            if word_layout:
                prelayout = []
                for block in blocks:
                    if block.translation_rich or _has_explicit_line_breaks(block.translation):
                        continue
                    select_region(block, config)
                    _apply_default_english_case_preferences(block, config)
                    if not _apply_default_english_line_break_method(
                            block, block.original_font_size, None, config):
                        prelayout.append(block)
                if prelayout:
                    resize_unscaled(cleaned, prelayout, config)
            rendered = cleaned.copy()
            if free:
                points = fit_source_boxes(cleaned, free, free_texts, config, [b.min_rect for b in enclosed])
                rendered = asyncio.run(dispatch(rendered, free, config, None,
                    skip_font_scaling=True, skip_text_replacements=True, layout_points=points))
            if enclosed:
                config.render.layout_mode = 'balloon_fill'
                config.render.balloon_fill_mask_layout = word_layout
                unscaled_points = None
                if rewrap:
                    # Feed the fitted size back to MTU's word wrapper once.
                    unscaled_points = resize_unscaled(rendered, enclosed, config, original, bubble_mask)
                    for block, text in zip(enclosed, enclosed_texts):
                        previous_breaks = block.translation
                        block.translation = text
                        select_region(block, config)
                        _apply_default_english_line_break_method(
                            block, round(block.font_size), None, config)
                        if block.translation != previous_breaks:
                            unscaled_points = None  # A changed wrap also invalidates its neighbours' fit.
                enclosed, points = fit_bubbles(rendered, original, enclosed, enclosed_texts, config,
                                               bubble_mask, labels, [block.dst_points for block in free],
                                               language=language, unscaled_points=unscaled_points)
                rendered = asyncio.run(dispatch(rendered, enclosed, config, original,
                    bubble_mask=bubble_mask, skip_font_scaling=True, skip_text_replacements=True,
                    layout_points=points))
            return rendered


class Engine:
    def __init__(self, models, font, gpu=0, threads=2, detect_size=1280, inpainting_size=512, keep_lang=None,
                 ocr_batch_size=None, detect_batch_size=None, backend='cuda', ocr_model=None, detect_precision='auto'):
        from classic_node.config import ENGINE_DEFAULTS
        import cv2
        import torch
        validate_backend(backend, gpu)
        self.models = Path(models)
        self.gpu = gpu
        self.backend = backend
        self.device = 'mps' if backend == 'mps' else f'cuda:{gpu}'
        self.keep_lang = keep_lang
        self.renderer = Renderer(models, font)
        self.config = configuration(detect_size, inpainting_size)
        # Detection wrappers own mutable state. OCR adapters protect their own
        # model calls, without serializing other models or CPU preparation.
        self._detector_lock = Lock()
        self._bubble_lock = Lock()
        torch.set_num_threads(threads)
        if backend != 'mps':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        from manga_translator.utils import ModelWrapper, MangaLensBubbleDetector
        ModelWrapper._MODEL_DIR = str(self.models)
        from manga_translator.detection.default import DefaultDetector
        from ballontranslator.utils.logger import logger as ballons_logger
        ballons_logger.handlers = [logging.NullHandler()]
        ballons_logger.propagate = False
        ballons_logger.setLevel(logging.CRITICAL + 1)
        from manga_translator.inpainting.inpainting_lama_mpe import LamaLargeInpainter
        from .ocr.hayai import HayaiOCR
        # Select only the upstream PyTorch checkpoint; do not fetch an unused CPU
        # ONNX model just to satisfy the upstream multi-backend download map.
        class TorchLama(LamaLargeInpainter):
            _MODEL_MAPPING = {'model': LamaLargeInpainter._MODEL_MAPPING['model']}
        with device_context(backend, gpu):
            self.detector = DefaultDetector()
            self.detector.max_batch_size = (ENGINE_DEFAULTS['detect_batch_size']
                                           if detect_batch_size is None else detect_batch_size)
            self.inpainter = TorchLama()
            for model in (self.detector, self.inpainter):
                # Keep upstream model diagnostics out of operational logs.
                model.logger.handlers = [logging.NullHandler()]
                model.logger.propagate = False
                model.logger.setLevel(logging.CRITICAL + 1)
                if not model.is_downloaded():
                    raise ValueError('Missing prepared model; runtime downloads are disabled')
                device = 'mps' if backend == 'mps' else 'cuda'
                if model is self.detector:
                    asyncio.run(model.load(device, precision=detect_precision))
                elif backend != 'cuda':
                    asyncio.run(model.load(device, force_torch=True))
                else:
                    asyncio.run(model.load(device))
            self.detector.model.register_forward_pre_hook(check_cancelled)
            self.detector_settings = {'dtype': str(next(self.detector.model.parameters()).dtype),
                                      'opencl_filter': self.detector.opencl_filter}
            self.bubbles = MangaLensBubbleDetector(
                model_path=self.models / 'detection/mangalens.pt', device=self.device,
                imgsz=768, conf=0.25, iou=0.7, auto_download=False, auto_load=True)
            self.ocr = HayaiOCR(ocr_model or self.models / 'hayai', gpu, backend=backend,
                                batch_size=ENGINE_DEFAULTS['ocr_batch_size'] if ocr_batch_size is None else ocr_batch_size)
        self._capture_guard = self.ocr.capture_lock or nullcontext()
        if backend in ('cuda', 'amd'):
            from .inpainting import LamaGraphs
            self.inpainter.model.generator = LamaGraphs(self.inpainter.model.generator)
        # Upstream imports can disable OpenCV threading. Apply the node's CPU
        # budget after loading them, including DBNet and mask bilateral filters.
        cv2.setNumThreads(threads)

    def warmup(self):
        rgb = np.full((512, 512, 3), 255, dtype=np.uint8)
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[230:260, 230:260] = 255
        async def run():
            c = self.config
            await self.detector.detect(rgb, c.detector.detection_size, c.detector.text_threshold,
                                       c.detector.box_threshold, c.detector.unclip_ratio)
            await self.ocr.warmup(rgb, c.ocr)
            await self.inpainter.inpaint(rgb, mask, c.inpainter, c.inpainter.inpainting_size)
            if self.backend == 'amd':
                # Native per-block LaMa inputs are square and padded to 64.
                # Warm their HIP kernels before accepting pages, not on first use.
                for side in range(64, c.inpainter.inpainting_size + 1, 64):
                    if side == 512:
                        continue  # Already exercised above.
                    sample = np.full((side, side, 3), 255, dtype=np.uint8)
                    sample_mask = np.zeros((side, side), dtype=np.uint8)
                    center = side // 2
                    sample_mask[center - 15:center + 15, center - 15:center + 15] = 255
                    await self.inpainter.inpaint(sample, sample_mask, c.inpainter,
                                                c.inpainter.inpainting_size)
            if self.backend in ('cuda', 'amd'):
                self.inpainter.model.generator.prepare()
        with device_context(self.backend, self.gpu), self._detector_lock, self._bubble_lock:
            asyncio.run(run())
            self.bubbles.detect(rgb, device=self.device)

    def analyze(self, rgb):
        from manga_translator.mask_refinement import dispatch as refine
        from ballontranslator.utils.textblock import group_output
        from manga_translator.utils import (
            TextBlock, build_bubble_mask_from_mangalens_result, is_valuable_text)
        async def run():
            check_cancelled()
            c = self.config
            with stage('analyze_detect'), waiting_for(self._detector_lock, 'detect_lock_wait'), self._capture_guard:
                # Pack long-page stripes without shrinking below their source
                # short side; native overlap/merging and detect_size still apply.
                lines, raw, _ = await self.detector.detect(rgb, c.detector.detection_size,
                    c.detector.text_threshold, c.detector.box_threshold, c.detector.unclip_ratio,
                    det_rearrange_min_effective_short_side=min(rgb.shape[:2]))
            check_cancelled()
            # Retain the approved upstream paragraph grouping and reading
            # order, using DBNet lines without loading or running a CTD model.
            with stage('analyze_group'):
                blocks = group_output([], [line.pts for line in lines], rgb.shape[1], rgb.shape[0])
            readings = await self.ocr.recognize(rgb, blocks, c.ocr)
            check_cancelled()
            regions = []
            for block, reading in zip(blocks, readings, strict=True):
                if not reading.texts:
                    continue
                region = TextBlock(lines=block.lines, texts=reading.texts, language=None,
                    font_size=block.font_size, angle=block.angle,
                    direction='v' if block.src_is_vertical else 'h',
                    default_stroke_width=c.render.stroke_width, adjust_bg_color=False)
                regions.append(region)
            regions = [region for region in regions
                       if len(region.text.strip()) >= c.ocr.min_text_length and is_valuable_text(region.text)
                       and (self.keep_lang is None or region.source_lang == self.keep_lang)]
            if not regions:
                return [], None, None, None
            with stage('analyze_colors'):
                sample_colors(rgb, regions, raw)
            with stage('analyze_bubbles'), self._bubble_lock, self._capture_guard:
                check_cancelled()
                detected_bubbles = self.bubbles.detect(rgb, device=self.device)
            with stage('analyze_bubble_mask'):
                # Native mask construction still reads GPU tensors. Association
                # only consumes the resulting CPU mask and independent regions.
                with self._capture_guard:
                    bubbles = build_bubble_mask_from_mangalens_result(detected_bubbles, rgb.shape[:2])
                associate_bubbles(regions, bubbles)
            with stage('analyze_refine'):
                check_cancelled()
                mask = await refine(regions, rgb, raw, dilation_offset=c.mask_dilation_offset,
                                    kernel_size=c.kernel_size, limit_mask_dilation_to_bubble_mask=True,
                                    bubble_mask=bubbles)
            return regions, mask, raw, bubbles
        with device_context(self.backend, self.gpu):
            return asyncio.run(run())

    def inpaint(self, rgb, mask, raw_mask, bubble_mask, regions, *, cache=None):
        check_cancelled()
        import cv2
        from manga_translator.utils import TextBlock, erode_bubble_mask
        from manga_translator.inpainting.ballon_fill import (
            MODEL_BUBBLE_SHRINK_RATIO, inpaint_regions_per_block, solid_fill_pure_bubbles)
        # These are MTU's documented stage inputs, including its 2px raw-mask
        # dilation for background sampling; no local repair/layout algorithm.
        blocks = [TextBlock(**data) for data in regions]
        with stage('inpaint_prepare'):
            tight = raw_mask
            if tight.shape[:2] != rgb.shape[:2]:
                tight = cv2.resize(tight, rgb.shape[1::-1], interpolation=cv2.INTER_LINEAR)
            tight = cv2.dilate(cv2.compare(tight, 127, cv2.CMP_GE), None, iterations=2)
            inset = erode_bubble_mask(bubble_mask, MODEL_BUBBLE_SHRINK_RATIO)
            components = None
            if np.any(inset):
                _, labels, stats, _ = cv2.connectedComponentsWithStats(inset, connectivity=8)
                components = (labels, stats)
            if cache is not None:
                inset.flags.writeable = False
                cache['bubble_inset'] = inset
                # Persistent page masks have a 6P allowance. Pathological tiny
                # components must not retain an unbounded statistics table.
                if components is not None and stats.nbytes <= inset.nbytes:
                    labels.flags.writeable = stats.flags.writeable = False
                    cache['bubble_labels'], cache['bubble_stats'] = components
        with stage('inpaint_solid_fill'):
            check_cancelled()
            filled, remaining, solid_regions = solid_fill_pure_bubbles(rgb, mask, blocks, tight,
                inset, self.config.ocr.model_bubble_overlap_threshold, components=components)
            record('inpaint_solid_regions', solid_regions)
        async def inpaint(crop, local_mask):
            check_cancelled()
            # The loaded FP32 LaMa eval path only reads model state; its input
            # tensors and padding are per-call. The compute pool bounds readers.
            started = perf_counter()
            with self._capture_guard:
                check_cancelled()
                record('inpaint_lock_wait', perf_counter() - started)
                with stage('inpaint_model'):
                    return await self.inpainter.inpaint(crop, local_mask, self.config.inpainter,
                                                        self.config.inpainter.inpainting_size)
        with stage('inpaint_lama'), device_context(self.backend, self.gpu):
            check_cancelled()
            # solid_fill returns an owned mask; the native block loop may clear it.
            result, count = asyncio.run(inpaint_regions_per_block(filled, remaining, inpaint))
            record('inpaint_blocks', count)
        return result

    def close(self):
        # Node shutdown joins the compute pool before unloading model weights.
        with device_context(self.backend, self.gpu), self._detector_lock, self._bubble_lock:
            if self.backend == 'cuda' and self.inpainter.model.generator.finished is not None:
                self.inpainter.model.generator.finished.synchronize()
            self.ocr.close()
            for model in (self.detector, self.inpainter):
                asyncio.run(model.unload())
            self.bubbles.model = None
