"""Per-operation node diagnostics; never part of image or analysis identity."""
from contextlib import contextmanager
from contextvars import ContextVar
import math
from threading import Lock
from time import perf_counter


current = ContextVar('image_timings', default=None)

# The deployed v3 center rejects unknown timing keys. Detailed local diagnostics
# must not silently extend that wire contract (or become image identity).
WIRE_FIELDS = frozenset(('download', 'download_queue', 'analyze', 'analyze_queue',
    'inpaint', 'inpaint_queue', 'render', 'render_queue', 'analysis_submit', 'text_wait',
    'local_total', 'output_put', 'render_areas', 'render_layout', 'render_diff',
    'render_encode', 'detect_lock_wait', 'ocr_lock_wait', 'inpaint_lock_wait'))

OCR_FIELDS = frozenset(('analyze_detect', 'detect_filter', 'detect_filter_opencl_chunks', 'detect_filter_cpu_pages',
    'analyze_group', 'analyze_ocr',
    'analyze_colors', 'analyze_bubbles', 'analyze_bubble_mask', 'analyze_refine', 'analyze_checkpoint',
    'analyze_color_gate', 'color_fast_regions', 'color_model_lines', 'color_batches', 'color_batch_max',
    'ocr_crop', 'ocr_preprocess', 'ocr_transfer', 'ocr_infer', 'ocr_decode', 'ocr_lock_wait', 'ocr_model_lock_wait',
    'ocr_paragraphs', 'ocr_batches', 'ocr_batch_max', 'ocr_output_tokens', 'ocr_truncated',
    'ocr_graph_batches', 'ocr_eager_batches', 'ocr_capture_wait', 'ocr_capture', 'ocr_graph_captures'))

LOCAL_TIMINGS = WIRE_FIELDS | OCR_FIELDS | frozenset((
    'prepare', 'prepare_queue', 'package', 'package_queue', 'persist', 'persist_queue',
    'deliver', 'deliver_queue', 'download_sign', 'download_sign_attempts',
    'download_get', 'download_get_attempts', 'download_retries',
    'upload_sign', 'upload_sign_attempts', 'upload_put', 'upload_put_attempts',
    'upload_retries', 'complete', 'complete_attempts', 'complete_retries',
    'inpaint_prepare', 'inpaint_solid_fill', 'inpaint_lama', 'inpaint_model',
    'inpaint_blocks', 'inpaint_solid_regions', 'inpaint_graph_blocks', 'inpaint_eager_blocks'))


def safe_timings(values):
    return {name: round(value, 6) for name, value in values.items() if name in LOCAL_TIMINGS
            and type(value) in (int, float) and math.isfinite(value) and value >= 0}


@contextmanager
def collect(fields=None):
    """Collect all offline metrics, or only the requested production fields."""
    values, lock = {}, Lock()
    token = current.set((values, lock, fields))
    try:
        yield values
    finally:
        current.reset(token)


def _target(name):
    target = current.get()
    if target is not None and (target[2] is None or name in target[2]):
        return target


def record(name, seconds, *, maximum=False):
    target = _target(name)
    if target is not None:
        values, lock, _ = target
        with lock:
            values[name] = max(values.get(name, 0.), seconds) if maximum else values.get(name, 0.) + seconds


@contextmanager
def stage(name):
    """Wall time, including any native synchronization inside the operation."""
    if _target(name) is None:
        yield
        return
    start = perf_counter()
    try:
        yield
    finally:
        record(name, perf_counter() - start)


@contextmanager
def waiting_for(lock, *names):
    start = perf_counter() if any(_target(name) is not None for name in names) else None
    lock.acquire()
    try:
        if start is not None:
            elapsed = perf_counter() - start
            for name in names:
                record(name, elapsed)
        yield
    finally:
        lock.release()
