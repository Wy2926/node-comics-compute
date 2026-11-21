"""Local Hayai paragraph recognition using the pinned upstream inference engine."""
import json
from contextlib import contextmanager
from pathlib import Path
import sys
from threading import Condition, Lock

from PIL import Image

from classic_node.timing import record, stage, waiting_for
from mtu_engine.backend import CUDA_GRAPH_MIN_FREE, check_cancelled, device_context
from . import Reading
from .crops import line_context_boxes, paragraph_box, paragraph_crop
from .models import verify_hayai

MAX_PATCHES = 256
MAX_TOKENS = 512


class CudaModelCalls:
    """Independent models can overlap; graph capture must drain all of them."""
    def __init__(self):
        self.admission = Lock()
        self.idle = Condition()
        self.active = 0

    def __enter__(self):
        with self.admission, self.idle:
            self.active += 1

    def __exit__(self, *_):
        with self.idle:
            self.active -= 1
            if not self.active:
                self.idle.notify_all()

    @contextmanager
    def capture(self):
        with waiting_for(self.admission, 'ocr_capture_wait'):
            with self.idle, stage('ocr_capture_wait'):
                self.idle.wait_for(lambda: not self.active)
            yield


def complete_texts(sequences, tokenizer):
    """A capped paragraph must not authorize erasing text it did not finish reading."""
    stops = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    texts = []
    for sequence in sequences:
        end = next((i for i, token in enumerate(sequence[1:], 1) if token in stops), None)
        record('ocr_output_tokens', (end - 1) if end is not None else len(sequence) - 1)
        record('ocr_truncated', int(end is None))
        texts.append(tokenizer.decode(sequence[1:end], skip_special_tokens=True).strip()
                     if end is not None else '')
    return texts


class HayaiOCR:
    def __init__(self, model, gpu=0, *, backend='cuda', batch_size=4):
        if type(batch_size) is not int or not 1 <= batch_size <= 8:
            raise ValueError('Hayai batch_size must be between 1 and 8')
        import torch
        from transformers import AutoConfig, AutoImageProcessor, AutoModel, PreTrainedTokenizerFast

        root = Path(model).resolve()
        self.identity = verify_hayai(root) | {'max_num_patches': MAX_PATCHES, 'max_new_tokens': MAX_TOKENS}
        self.backend, self.gpu, self.batch_size = backend, gpu, batch_size
        self.device = 'mps' if backend == 'mps' else f'cuda:{gpu}'
        self._lock = Lock()
        with device_context(backend, gpu):
            config = AutoConfig.from_pretrained(root / 'model', trust_remote_code=True, local_files_only=True)
            config.vision_config = json.loads((root / 'vision/config.json').read_text())['vision_config']
            model, loading = AutoModel.from_pretrained(root / 'model', config=config,
                trust_remote_code=True, local_files_only=True, dtype=torch.float32, output_loading_info=True)
            if any(loading[key] for key in ('missing_keys', 'mismatched_keys', 'unexpected_keys', 'error_msgs')):
                raise ValueError('Hayai checkpoint does not match its pinned inference code')
            model.drop_ids_head()
            model = model.to(self.device).eval()
            self.processor = AutoImageProcessor.from_pretrained(root / 'model', trust_remote_code=True,
                local_files_only=True, use_fast=False, num_workers=1)
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(root / 'model', local_files_only=True)
            # The pinned upstream engine exposes raw tokens and owns its fused
            # weights/KV cache. Avoid generate()'s silent FP32/reference fallback
            # and retain EOS information for truncation detection.
            self.engine = sys.modules[type(model).__module__]._FastEngine(model, torch.float16)
            self.engine.MAX_STATES = 4  # Supernova caches eager KV states as well as graphs.
            free = torch.cuda.mem_get_info(gpu)[0] if backend != 'mps' else 0
            self.capture_lock = Lock() if backend == 'amd' else (
                CudaModelCalls() if free >= CUDA_GRAPH_MIN_FREE else None)
            self.identity['decoder_graph'] = self.capture_lock is not None
            if self.capture_lock is not None:
                self.engine._capture = self._capture
                if free >= 8 * 1024**3:
                    # Batch 8 can use all eight row counts plus a second visual-prefix bucket.
                    self.engine.MAX_STATES = 9 if batch_size == 8 else 8
                elif backend == 'cuda':
                    # The fifth prefix/batch shape avoids recapture on the 8 GiB RTX 4060.
                    self.engine.MAX_STATES = 5

    def _capture(self, state):
        # Device synchronization from another model can invalidate capture.
        # Replay uses the ordinary OCR model lock, not this cross-model guard.
        guard = (self.capture_lock.capture() if self.backend == 'cuda'
                 else waiting_for(self.capture_lock, 'ocr_capture_wait'))
        with guard, stage('ocr_capture'):
            result = type(self.engine)._capture(self.engine, state)
            record('ocr_graph_captures', 1)
            return result

    async def recognize(self, image, blocks, config):
        height, width = image.shape[:2]
        boxes = [paragraph_box(block.xyxy, width, height) for block in blocks]
        indices = sorted((i for i, (l, t, r, b) in enumerate(boxes)
                          if r > l and b > t and len(blocks[i].lines)),
                         key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
        boxes = line_context_boxes(boxes, blocks, indices, width, height)
        results = [Reading([]) for _ in blocks]
        record('ocr_paragraphs', len(indices))
        with stage('analyze_ocr'):
            for offset in range(0, len(indices), self.batch_size):
                check_cancelled()
                owners = indices[offset:offset + self.batch_size]
                with stage('ocr_crop'):
                    crops = [paragraph_crop(image, i, boxes, blocks, indices) for i in owners]
                for i, text in zip(owners, self._recognize(crops), strict=True):
                    results[i] = Reading([text] + [''] * (len(blocks[i].lines) - 1))
        return results

    def _recognize(self, images):
        import torch

        record('ocr_batches', 1)
        record('ocr_batch_max', len(images), maximum=True)
        with stage('ocr_preprocess'):
            inputs = self.processor(images=images, max_patches=MAX_PATCHES, return_tensors='pt')
            if inputs.pop('valid_indices') != list(range(len(images))):
                raise ValueError('Hayai processor skipped or reordered paragraph crops')
        with device_context(self.backend, self.gpu), waiting_for(
                self._lock, 'ocr_lock_wait', 'ocr_model_lock_wait'), torch.inference_mode():
            check_cancelled()
            with stage('ocr_transfer'):
                # Native FastEngine derives the mask from CPU spatial_shapes.
                # Moving that metadata to GPU only makes run() copy it back.
                inputs['pixel_values'] = inputs['pixel_values'].to(self.device)
            with stage('ocr_infer'):
                tokens = self.engine.run(**inputs, bos=self.tokenizer.bos_token_id,
                    eos=self.tokenizer.eos_token_id, pad=self.tokenizer.pad_token_id,
                    max_new=MAX_TOKENS, use_graph=self.capture_lock is not None)
            record('ocr_graph_batches' if self.capture_lock is not None and not self.engine._graph_failed
                   else 'ocr_eager_batches', 1)
        with stage('ocr_decode'):
            return complete_texts(tokens.tolist(), self.tokenizer)

    async def warmup(self, image, config):
        self._recognize([Image.fromarray(image[:96, :192])])

    def close(self):
        with device_context(self.backend, self.gpu), self._lock:
            self.engine = self.processor = self.tokenizer = None
