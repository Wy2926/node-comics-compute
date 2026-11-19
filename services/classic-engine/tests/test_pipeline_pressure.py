import pytest
from tools.validate_pipeline_pressure import exercise, FixtureRuntime


@pytest.mark.parametrize('scenario', ['text', 'upload', 'mixed'])
@pytest.mark.parametrize('local_pages', [1, 8])
def test_eight_leases_four_uploads_keep_making_progress(tmp_path, scenario, local_pages):
    result = exercise(tmp_path, scenario, local_pages=local_pages, timeout=30)
    assert result['completed'] == 24 and result['journal_empty']
    assert sum(result['claim_batches']) == 24 and max(result['claim_batches']) <= 4
    if scenario == 'upload':
        assert result['network_peak']['upload'] == 4
    if scenario == 'mixed':
        assert result['blocked_snapshot']['completed'] == 19
        spans = result['stage_spans']
        network = [span for span in spans if span['stage'] in {'download', 'analysis_submit', 'deliver'}]
        compute = [span for span in spans if span['stage'] in {'analyze', 'inpaint', 'render'}]
        assert any(left['lease_id'] != right['lease_id'] and left['start_s'] < right['end_s']
                   and right['start_s'] < left['end_s'] for left in network for right in compute)


@pytest.mark.parametrize('empty_ocr', [False, True])
def test_throughput_accepts_unchanged_pages_without_upload_or_duplicate_settlement(tmp_path, empty_ocr):
    from classic_node.protocol import pack_result
    class OriginalRuntime(FixtureRuntime):
        def analyze(self, *args, **kwargs):
            analysis = super().analyze(*args, **kwargs)
            if empty_ocr:
                analysis.update(segments=[], regions=[], mask=None)
            return analysis
        def inpaint(self, *args, **kwargs):
            assert not empty_ocr, 'An empty OCR page must skip inpainting'
            return super().inpaint(*args, **kwargs)
        def render(self, original, cleaned, analysis, translated, language, alpha, **kwargs):
            if empty_ocr:
                assert cleaned is original
            return pack_result(original, original, alpha, self.version, analysis, translated)
    result = exercise(tmp_path, 'throughput', OriginalRuntime(), total=8, timeout=15)
    assert result['completed'] == 8 and result['journal_empty']
    assert all(page['output_bytes'] == 0 for page in result['page_profiles'])
    assert result['network_peak'].get('upload', 0) == 0
    if empty_ocr:
        assert all('inpaint' not in page['timings'] for page in result['page_profiles'])


def test_measurement_fails_immediately_on_a_stage_error(tmp_path):
    class BrokenRuntime(FixtureRuntime):
        def inpaint(self, *args, **kwargs):
            raise RuntimeError('private fixture details')
    with pytest.raises(AssertionError, match='inpaint') as failure:
        exercise(tmp_path, 'throughput', BrokenRuntime(), total=8, timeout=15)
    assert 'private fixture details' not in str(failure.value)


def test_offline_profiles_keep_detailed_metrics_with_production_filter(tmp_path):
    from classic_node.timing import record, stage
    class ProfiledRuntime(FixtureRuntime):
        def analyze(self, *args, **kwargs):
            record('ocr_batches', 2)
            record('ocr_batch_max', 4, maximum=True)
            with stage('ocr_infer'):
                return super().analyze(*args, **kwargs)
    result = exercise(tmp_path, 'throughput', ProfiledRuntime(), total=8, timeout=15)
    assert result['completed'] == 8 and result['journal_empty']
    for profile in result['page_profiles']:
        assert profile['timings']['ocr_batches'] == 2
        assert profile['timings']['ocr_batch_max'] == 4
        assert profile['timings']['ocr_infer'] >= 0


@pytest.mark.parametrize('pipeline', [False, True])
def test_mps_page_benchmark_uses_metal_memory_and_real_dispatch_limit(tmp_path, monkeypatch, pipeline):
    import json
    from types import SimpleNamespace
    from PIL import Image
    from tools import benchmark_pages as bench
    inputs = tmp_path / 'input'
    inputs.mkdir()
    Image.new('RGB', (32, 32)).save(inputs / 'p.png')
    (tmp_path / 'licenses').mkdir()
    (tmp_path / 'licenses/font-sources.json').write_text('{"fonts":[]}')
    monkeypatch.setattr(bench, 'torch', SimpleNamespace(__version__='fixture', version=SimpleNamespace(cuda=None, hip=None),
        backends=SimpleNamespace(mps=SimpleNamespace(get_name=lambda: 'Apple GPU')),
        mps=SimpleNamespace(current_allocated_memory=lambda: 123)))  # No CUDA API in this fixture.
    runtime = FixtureRuntime()
    runtime.warmup = runtime.close = lambda: None
    runtime.engine = SimpleNamespace(gpu=0, detector_settings={}, ocr=SimpleNamespace(identity={}))
    monkeypatch.setattr(bench, 'Runtime', lambda _: runtime)
    args = SimpleNamespace(input=inputs, output=tmp_path/'result.json', models=tmp_path/'models', backend='mps',
        detect_precision='auto', auto_cpu=False, threads=2, render_workers=1, cache_mib=0, hayai_model=None,
        concurrency=2, pipeline=pipeline, total=8, resident_mib=512, repeats=1, reserve_vram_mib=1, inpaint_slots=3)
    with pytest.raises(ValueError, match='--inpaint-slots'):
        bench.run(args)
    args.inpaint_slots = 1 if pipeline else None
    with pytest.raises(ValueError, match='CUDA or HIP'):
        bench.run(args)
    args.reserve_vram_mib = 0
    bench.run(args)
    report = json.loads(args.output.read_bytes())
    row, = report['runs']
    assert report['gpu'] == 'Apple GPU' and row['mps_current_allocated_bytes'] == 123
    assert 'torch_peak_allocated_bytes' not in row
    if pipeline:
        assert report['pipeline_inpaint_slots'] == 1
        assert row['completed'] == 8 and row['journal_empty']
        spans = sorted((s['start_s'], s['end_s']) for s in row['stage_spans'] if s['stage'] in {'analyze', 'inpaint'})
        assert len(spans) == 16 and all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    else:
        assert row['pages'][0]['regions'] == 1
