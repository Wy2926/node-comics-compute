from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import os
import sys
from threading import Event, Lock
from types import SimpleNamespace

import numpy as np
import pytest

from classic_node.timing import collect
from mtu_engine.ocr.hayai import HayaiOCR, MAX_PATCHES, MAX_TOKENS


def test_cancelled_hayai_stops_before_next_crop_batch():
    import asyncio
    from classic_node.protocol import NodeFailure
    from mtu_engine.backend import cancellation
    ocr = HayaiOCR.__new__(HayaiOCR)
    ocr.batch_size = 2
    calls = []
    ocr._recognize = lambda crops: calls.append(len(crops)) or ['read'] * len(crops)
    blocks = [SimpleNamespace(xyxy=(0, 0, 20, 20), lines=[None]) for _ in range(5)]
    def check():
        if calls:
            raise NodeFailure('LEASE_STOPPED')
    with pytest.raises(NodeFailure, match='LEASE_STOPPED'), cancellation(check):
        asyncio.run(ocr.recognize(np.zeros((24, 24, 3), np.uint8), blocks, None))
    assert calls == [2]
    assert len(asyncio.run(ocr.recognize(np.zeros((24, 24, 3), np.uint8), blocks, None))) == 5
    assert calls == [2, 2, 2, 1]


@pytest.mark.parametrize('fail_first', [False, True])
@pytest.mark.parametrize('backend,graph_failed', [('amd', False), ('amd', True), ('cuda', False)])
def test_hayai_prepares_next_page_outside_model_lock_and_keeps_results_local(monkeypatch, fail_first,
                                                                         backend, graph_failed):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        inference_mode=nullcontext, cuda=SimpleNamespace(device=lambda _: nullcontext())))
    entered, prepared, attempted, release = Event(), Event(), Event(), Event()
    lock = Lock()
    calls = []
    ocr = HayaiOCR.__new__(HayaiOCR)
    ocr.backend, ocr.gpu, ocr.device = backend, 0, 'cuda:0'
    def acquire():
        if lock.locked():
            attempted.set()
        lock.acquire()
    ocr._lock = SimpleNamespace(acquire=acquire, release=lock.release)

    class Tensor:
        def __init__(self, value, kind):
            self.value, self.kind = value, kind
        def to(self, device):
            assert lock.locked() and device == 'cuda:0'
            assert self.kind == 'pixels', 'Native metadata should stay on CPU'
            return self

    def processor(*, images, max_num_patches, return_tensors):
        assert max_num_patches == MAX_PATCHES and return_tensors == 'pt'
        if images[0] == 22:
            assert lock.locked()
            prepared.set()
        return {'pixel_values': Tensor(images[0], 'pixels'),
                'pixel_attention_mask': Tensor(1, 'mask'), 'spatial_shapes': Tensor(2, 'shapes')}

    def run(**kwargs):
        assert lock.locked()
        assert kwargs['max_new'] == MAX_TOKENS
        assert kwargs['use_graph'] is (backend == 'amd')
        value = kwargs['pixel_values'].value
        calls.append(value)
        if value == 11:
            entered.set()
            assert release.wait(5)
            if fail_first:
                raise RuntimeError('fixture model failure')
        return np.array([[1, value, 2, 0]])

    def decode(tokens, **kwargs):
        return str(tokens[0])

    ocr.processor, ocr.engine = processor, SimpleNamespace(run=run, _graph_failed=graph_failed)
    ocr.tokenizer = SimpleNamespace(bos_token_id=1, eos_token_id=2, pad_token_id=0, decode=decode)
    def recognize(value):
        with collect() as timings:
            text = ocr._recognize([value])
        return text, timings

    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(recognize, 11)
        try:
            assert entered.wait(5)
            second = pool.submit(recognize, 22)
            assert prepared.wait(5) and attempted.wait(5)
            assert calls == [11]
        finally:
            release.set()
        if fail_first:
            with pytest.raises(RuntimeError, match='fixture model failure'):
                first.result(timeout=5)
        else:
            assert first.result(timeout=5)[0] == ['11']
        text, timings = second.result(timeout=5)
    assert text == ['22'] and calls == [11, 22] and not lock.locked()
    assert timings['ocr_batch_max'] == timings['ocr_batches'] == timings['ocr_output_tokens'] == 1
    assert timings['ocr_truncated'] == 0
    assert timings['ocr_graph_batches' if backend == 'amd' and not graph_failed else 'ocr_eager_batches'] == 1
    assert timings['ocr_model_lock_wait'] == timings['ocr_lock_wait']
    assert all(timings[name] >= 0 for name in ('ocr_preprocess', 'ocr_transfer', 'ocr_infer', 'ocr_decode'))


@pytest.mark.parametrize('fail', [False, True])
def test_graph_capture_waits_for_other_gpu_stage_and_releases_on_failure(fail):
    ocr = HayaiOCR.__new__(HayaiOCR)
    lock = Lock()
    entered, attempted = Event(), Event()
    def acquire():
        attempted.set()
        lock.acquire()
    ocr.capture_lock = SimpleNamespace(acquire=acquire, release=lock.release)
    class Native:
        def _capture(self, state):
            assert lock.locked() and state == 'fixture'
            entered.set()
            if fail:
                raise RuntimeError('capture failed')
            return 'native graph'
    ocr.engine = Native()
    ocr.engine._capture = ocr._capture
    with ThreadPoolExecutor(1) as pool:
        with lock:
            future = pool.submit(ocr.engine._capture, 'fixture')
            assert attempted.wait(5)
            assert not entered.is_set()
        if fail:
            with pytest.raises(RuntimeError, match='capture failed'):
                future.result(timeout=5)
        else:
            assert future.result(timeout=5) == 'native graph'
    assert entered.is_set() and not lock.locked()


@pytest.mark.skipif(os.environ.get('MTU_TEST_BACKEND') != 'amd' or not os.environ.get('MTU_TEST_HAYAI'),
                    reason='Requires a prepared Hayai model and the AMD GPU environment')
def test_hip_graph_lru_retains_fifth_batch_size_only_with_headroom(monkeypatch):
    import torch
    from PIL import Image, ImageDraw
    free, total = torch.cuda.mem_get_info(0)
    if free < 10 * 1024**3:
        pytest.skip('Insufficient headroom to exercise both graph budgets')
    image = Image.new('RGB', (160, 48), 'white')
    ImageDraw.Draw(image).text((4, 4), 'HELLO WORLD', fill='black', font_size=22)
    outputs = []
    for available, captures, capacity in ((8 * 1024**3 - 1, 1, 4), (8 * 1024**3, 0, 9)):
        monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda *_, n=available: (n, total))
        ocr = HayaiOCR(os.environ['MTU_TEST_HAYAI'], backend='amd', batch_size=8)
        try:
            assert ocr.engine.MAX_STATES == capacity
            outputs.append([ocr._recognize([image] * n) for n in range(1, 6)])
            with collect() as timings:
                assert ocr._recognize([image]) == outputs[-1][0]
            assert timings['ocr_graph_batches'] == 1
            assert timings.get('ocr_graph_captures', 0) == captures
        finally:
            ocr.close()
    assert outputs[0] == outputs[1]
