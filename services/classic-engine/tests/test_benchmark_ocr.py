from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace
import json
import sys

from PIL import Image
import pytest

from tools import benchmark_ocr
from tools.benchmark_ocr import bounded_pages, summarize


def test_benchmark_replenishes_finished_pages_and_preserves_input_order():
    first, third, release = Event(), Event(), Event()
    lock = Lock()
    active = peak = 0

    def page(index):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            if index == 0:
                first.set()
                assert release.wait(5)
            elif index == 1:
                assert first.wait(5)
            elif index == 2:
                third.set()
                release.set()
            return index
        finally:
            with lock:
                active -= 1

    with ThreadPoolExecutor(2) as pool:
        try:
            assert bounded_pages(pool, page, list(range(8)), 2) == list(range(8))
        finally:
            release.set()
    assert third.is_set() and peak == 2 and active == 0


def test_summary_includes_zero_for_pages_that_skip_a_stage():
    result = summarize([{'timings': {'analyze': 2, 'analyze_ocr': 1}}, {'timings': {'analyze': 4}}])
    assert result['analyze']['mean'] == result['analyze']['p50'] == 3
    assert result['analyze_ocr']['mean'] == result['analyze_ocr']['p50'] == .5
    assert result['analyze']['p95'] == 3.9


@pytest.mark.parametrize('changed', [False, True])
def test_real_tool_report_omits_text_and_records_mismatch_before_closing(tmp_path, monkeypatch, changed):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(__version__='fixture',
        version=SimpleNamespace(hip=None, cuda=None)))
    source, output = tmp_path / 'page.png', tmp_path / 'report.json'
    Image.new('RGB', (8, 8)).save(source)
    calls, closed = [], []
    def analyze(rgb):
        calls.append(True)
        return [SimpleNamespace(text='private text' + (' changed' if changed and len(calls) > 1 else ''))], None, None, None
    runtime = SimpleNamespace(version='fixture', engine=SimpleNamespace(
        ocr=SimpleNamespace(identity={'backend': 'hayai'}), analyze=analyze), close=lambda: closed.append(True))
    monkeypatch.setattr(benchmark_ocr, 'load', lambda _: {
        'engine': {'ocr_backend': 'hayai', 'backend': 'mps'}, 'resident_bytes': 1024 ** 3,
        'node_token': 'private credential'})
    monkeypatch.setattr(benchmark_ocr, 'Runtime', lambda _: runtime)
    args = SimpleNamespace(config=tmp_path / 'unused.json', input=[source], output=output,
                           batch_size=4, concurrency=[1, 2], repeats=1, include_text=False)
    if changed:
        with pytest.raises(ValueError, match='OCR results changed'):
            benchmark_ocr.run(args)
    else:
        benchmark_ocr.run(args)
    report = json.loads(output.read_text(encoding='utf-8'))
    assert closed == [True] and 'private' not in output.read_text(encoding='utf-8')
    assert [run['same_text'] for run in report['runs']] == ([False] if changed else [True, True])
    assert report['runs'][0]['pages'][0]['regions'] == 1
    assert report['runs'][0]['pages'][0]['timings']['ocr_total'] == 0
