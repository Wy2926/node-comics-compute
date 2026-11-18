"""Target capabilities use Qt glyph coverage and the selected font fingerprint."""
from types import SimpleNamespace

import pytest

from classic_node import runtime


def prepare(monkeypatch, tmp_path, covers):
    font = tmp_path / 'font.fixture'
    font.write_bytes(b'font identity')
    monkeypatch.setattr(runtime, 'verify', lambda *args, **kwargs: {'files': {}})
    monkeypatch.setattr(runtime, 'package_version', lambda *args: 'test')
    monkeypatch.setattr(runtime, 'Engine', lambda **kwargs: SimpleNamespace(
        renderer=SimpleNamespace(covers=covers, font_ids=['comic-relief']),
        detector_settings={'dtype': 'torch.float32', 'opencl_filter': False},
        ocr=SimpleNamespace(identity={'backend': 'hayai'}), close=lambda: None))
    return {'engine': {'models': 'unused', 'gpu': 0, 'font': [str(font)]}}


def test_ascii_font_reports_only_renderable_targets(monkeypatch, tmp_path):
    config = prepare(monkeypatch, tmp_path, str.isascii)
    node = runtime.Runtime(config)
    assert node.languages == ['en', 'id']
    assert node.supported_font_ids == ['comic-relief']
    first = node.version
    (tmp_path / 'font.fixture').write_bytes(b'changed font')
    assert runtime.Runtime(config).version != first


def test_no_supported_font_rejects_before_claim(monkeypatch, tmp_path):
    config = prepare(monkeypatch, tmp_path, lambda text: False)
    with pytest.raises(ValueError, match='Fonts do not cover'):
        runtime.Runtime(config)


def test_language_default_mapping_changes_runtime_fingerprint(monkeypatch, tmp_path):
    config = prepare(monkeypatch, tmp_path, str.isascii)
    first = runtime.Runtime(config).version
    monkeypatch.setattr(runtime, 'DEFAULT_FONT_IDS', {**runtime.DEFAULT_FONT_IDS, 'en': 'patrick-hand'})
    assert runtime.Runtime(config).version != first


def test_gpu_backend_remains_part_of_runtime_identity(monkeypatch, tmp_path):
    config = prepare(monkeypatch, tmp_path, str.isascii)
    cuda = runtime.Runtime(config)
    assert cuda.version.startswith('mtu-cuda-v1-')
    config['engine']['backend'] = 'amd'
    amd = runtime.Runtime(config)
    assert amd.version.startswith('mtu-amd-v1-') and amd.version != cuda.version


def test_hayai_identity_needs_no_classic_packages_and_ignores_install_paths(monkeypatch, tmp_path):
    config = prepare(monkeypatch, tmp_path, str.isascii)
    config['engine'].update(backend='mps', ocr_model='weights')
    packages = []
    monkeypatch.setattr(runtime, 'package_version', lambda name: (packages.append(name), 'test')[1])
    first = runtime.Runtime(config).version
    assert not {'onnxruntime', 'onnxruntime-gpu', 'onnxruntime-directml', 'manga-ocr', 'rapidocr'} & set(packages)
    config['engine'].update(ocr_model='moved/weights')
    assert runtime.Runtime(config).version == first


def test_resolved_cpu_settings_start_only_cpu_pool_and_do_not_change_identity(monkeypatch, tmp_path):
    config = prepare(monkeypatch, tmp_path, str.isascii)
    config.update(render_workers=1, _render_threads=2)
    config['engine']['threads'] = 2
    direct = runtime.Runtime(config)
    assert direct.render_pool is None and direct.render_ipc_bytes(80, 64) == 0
    calls = []
    pool = SimpleNamespace(close=lambda: calls.append('closed'))
    monkeypatch.setattr(runtime, 'RenderPool', lambda *args: (calls.append(args), pool)[1])
    config.update(render_workers=4, _render_threads=1)
    config['engine']['threads'] = 3
    parallel = runtime.Runtime(config)
    assert parallel.version == direct.version and parallel.render_pool is pool
    assert calls == [('unused', config['engine']['font'], 4, 1)]
    parallel.close()
    assert calls[-1] == 'closed'
