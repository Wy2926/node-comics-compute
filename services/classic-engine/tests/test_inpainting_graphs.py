"""Graph lifecycle/fallback tests; only the explicit AMD test uses a GPU."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import os
from threading import Barrier
from types import SimpleNamespace

import pytest
import torch

from classic_node.timing import collect
from mtu_engine.inpainting import LamaGraphs, tensor_key


class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.25))
        self.broken = False

    def forward(self, image, mask, rel_pos=None, direct=None):
        if self.broken:
            raise RuntimeError('model/device failed')
        return image + mask * self.weight + (0 if rel_pos is None else rel_pos)


def test_unprepared_and_unsupported_inputs_remain_eager_without_gpu_initialization(monkeypatch):
    generator = Generator().eval()
    wrapper = LamaGraphs(generator)
    monkeypatch.setattr(wrapper, 'prepare', lambda: pytest.fail('Runtime must never capture graphs'))
    image, mask = torch.ones(1, 3, 20, 30), torch.ones(1, 1, 20, 30)
    with collect() as timings:
        for value in (None, 3):
            torch.testing.assert_close(wrapper(image, mask, value), generator(image, mask, value), rtol=0, atol=0)
    assert timings == {'inpaint_eager_blocks': 2}
    assert not wrapper.graphs and not wrapper.prepared
    assert tensor_key(image) != tensor_key(image.to(memory_format=torch.channels_last))
    assert tensor_key(image) != tensor_key(image.double())


@pytest.mark.parametrize('broken_device', [False, True])
def test_capture_failure_discards_partial_cache_and_validates_eager_fallback(monkeypatch, broken_device):
    generator = Generator().eval()
    wrapper = LamaGraphs(generator)
    stream = SimpleNamespace(wait_stream=lambda _: None, synchronize=lambda: None)
    calls = []
    def capture(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            generator.broken = broken_device
            raise RuntimeError('capture unsupported')
        return nullcontext()
    monkeypatch.setattr(torch.cuda, 'Stream', lambda **kwargs: stream)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda *args: stream)
    monkeypatch.setattr(torch.cuda, 'stream', lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, 'CUDAGraph', object)
    monkeypatch.setattr(torch.cuda, 'graph', capture)
    with pytest.raises(RuntimeError, match='model/device failed') if broken_device else nullcontext():
        wrapper.prepare()
    assert len(calls) == 2 and not wrapper.graphs
    if not broken_device:
        wrapper.prepare()
        assert len(calls) == 2  # No repeated capture attempts after fallback.
        torch.testing.assert_close(wrapper(torch.ones(1, 3, 4, 4), torch.ones(1, 1, 4, 4)),
                                   torch.full((1, 3, 4, 4), 1.25))


@pytest.mark.skipif(os.environ.get('MTU_TEST_BACKEND') != 'amd', reason='Requires explicit AMD GPU validation')
def test_hip_graph_outputs_are_owned_across_threads_streams_and_shapes():
    generator = Generator().cuda().eval()
    wrapper = LamaGraphs(generator)
    wrapper.prepare()
    assert len(wrapper.graphs) == 8
    original_graphs = tuple(wrapper.graphs)
    barrier = Barrier(2)
    def run(offset):
        stream = torch.cuda.Stream()
        results = []
        with torch.no_grad(), torch.cuda.stream(stream), collect() as timings:
            barrier.wait(timeout=10)
            for index, side in enumerate((64, 128, 512, 192, 64, 256)):
                image = torch.full((side, side, 3), offset + index / 16, device='cuda').permute(2, 0, 1).unsqueeze(0)
                mask = torch.ones((1, 1, side, side), device='cuda')
                results.append((wrapper(image, mask), generator(image, mask)))
            # Keep earlier outputs alive through later replays before reading them.
            for actual, expected in results:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert timings == {'inpaint_graph_blocks': 6}
        return results
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(run, (.125, .75)))
    with torch.inference_mode(), collect() as timings:
        image = torch.ones((576, 576, 3), device='cuda').permute(2, 0, 1).unsqueeze(0)
        mask = torch.ones((1, 1, 576, 576), device='cuda')
        torch.testing.assert_close(wrapper(image, mask), generator(image, mask), rtol=0, atol=0)
    assert timings == {'inpaint_eager_blocks': 1}
    assert tuple(wrapper.graphs) == original_graphs  # Runtime shapes cannot grow the cache.
