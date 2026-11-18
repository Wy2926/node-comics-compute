"""Color-model calls must not share mutable state between pages."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from classic_node.timing import collect
from mtu_engine.colors import _CpuColorModel


@pytest.mark.parametrize('fail_first', [False, True])
def test_color_model_serializes_calls_and_releases_lock_on_error(fail_first):
    entered, release = Event(), Event()
    def predict(value):
        if value == 1:
            entered.set()
            assert release.wait(5)
            if fail_first:
                raise ValueError('color failure')
        return [(value,)]
    model = _CpuColorModel(SimpleNamespace(infer_beam_batch_tensor=predict))
    def infer(value):
        with collect() as times:
            result = model.infer_beam_batch_tensor(value)
        return result, times['ocr_lock_wait']
    with ThreadPoolExecutor(2) as workers:
        first = workers.submit(infer, 1)
        try:
            assert entered.wait(5) and model._lock.locked()
            second = workers.submit(infer, 2)
        finally:
            release.set()
        if fail_first:
            with pytest.raises(ValueError, match='color failure'):
                first.result(timeout=5)
        else:
            assert first.result(timeout=5)[0] == [(1,)]
        result, waited = second.result(timeout=5)
    assert result == [(2,)] and waited >= 0 and not model._lock.locked()
