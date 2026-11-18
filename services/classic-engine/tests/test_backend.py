"""Backend selection is explicit; CUDA defaults and dependency closure stay isolated."""
import json
from pathlib import Path
import sys
import tomllib
from types import SimpleNamespace

import pytest

from classic_node.config import ENGINE_DEFAULTS, execution_settings, load
from mtu_engine.backend import validate_backend


@pytest.mark.parametrize('selection', [None, 'cuda', 'amd', 'mps'])
def test_backend_config_is_opt_in_and_keeps_resource_defaults(tmp_path, selection):
    engine = {} if selection is None else {'backend': selection}
    path = tmp_path / 'node.json'
    path.write_text(json.dumps({'protocol_version': 3, 'control_url': 'https://example.test',
        'node_id': 'test', 'node_token': 'fixture', 'resource_id': 'test', 'engine': engine}))
    config = load(path)
    assert config['engine']['backend'] == (selection or 'cuda')
    assert config['engine']['detect_precision'] == 'auto'
    assert config['local_pages'] == 2 and config['max_leases'] == 8
    assert config['inpaint_slots'] == 2
    assert config['engine']['ocr_batch_size'] == 4
    assert config['engine']['ocr_min_batch_size'] == 1
    assert config['engine']['ocr_arena_extend_strategy'] == 'kSameAsRequested'
    assert config['engine']['ocr_arena_shrinkage'] is True
    assert execution_settings(config)['engine']['backend'] == (selection or 'cuda')


@pytest.mark.parametrize('options,error', [({'backend': 'auto'}, 'backend must be cuda, amd or mps'),
                                          ({'detect_precision': 'fp8'}, 'detect_precision')])
def test_invalid_detector_or_device_choice_is_rejected(tmp_path, options, error):
    path = tmp_path / 'node.json'
    path.write_text(json.dumps({'protocol_version': 3, 'control_url': 'https://example.test',
        'node_id': 'test', 'node_token': 'fixture', 'resource_id': 'test', 'engine': options}))
    with pytest.raises(ValueError, match=error):
        load(path)


def test_detector_filter_respects_device_and_input_budget(monkeypatch):
    import numpy as np
    from mtu_engine import detection
    cv = detection.cv2
    props = SimpleNamespace(name='AMD Radeon', gcnArchName='gfx1030:sramecc-')
    cl = SimpleNamespace(name=lambda: 'gfx1030', type=lambda: cv.ocl.Device_TYPE_GPU, isAMD=lambda: True)
    for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(detection.torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(detection.torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(detection.torch.cuda, 'get_device_properties', lambda _: props)
    monkeypatch.setattr(cv.ocl, 'useOpenCL', lambda: True)
    monkeypatch.setattr(cv.ocl, 'Device_getDefault', lambda: cl)
    assert detection.matching_opencl_gpu('cuda')
    assert not detection.matching_opencl_gpu('cpu') and not detection.matching_opencl_gpu('mps')
    cl.name = lambda: 'another GPU'
    assert not detection.matching_opencl_gpu('cuda')
    cl.name = lambda: props.name
    monkeypatch.setattr(detection.torch.cuda, 'device_count', lambda: 2)
    assert not detection.matching_opencl_gpu('cuda')  # Two adapters with the same name.
    monkeypatch.setattr(detection.torch.cuda, 'get_device_properties',
                        lambda i: props if i == 0 else SimpleNamespace(name='another GPU'))
    assert detection.matching_opencl_gpu('cuda')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    assert not detection.matching_opencl_gpu('cuda')
    image = np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3)
    expected = cv.bilateralFilter(image, 17, 80, 80)
    monkeypatch.setattr(cv, 'UMat', lambda *args: pytest.fail('Unbudgeted device allocation'))
    for enabled, budget in ((False, 144), (True, 143)):
        np.testing.assert_array_equal(detection.filter_detector_image(image, enabled, budget), expected)


@pytest.mark.parametrize('backend,hip,platform,gpu,count,error', [
    ('cuda', None, 'win32', 0, 1, None),
    ('cuda', None, 'linux', 1, 2, None),
    ('amd', 'test-hip', 'win32', 0, 1, None),
    ('cuda', 'test-hip', 'win32', 0, 1, 'not ROCm'),
    ('amd', None, 'win32', 0, 1, 'separate Windows'),
    ('amd', 'test-hip', 'linux', 0, 1, 'separate Windows'),
    ('amd', 'test-hip', 'win32', 0, 2, 'single GPU'),
    ('amd', 'test-hip', 'win32', 1, 2, 'single GPU'),
    ('amd', 'test-hip', 'win32', 0, 0, 'CPU fallback is disabled'),
    ('cuda', None, 'win32', 1, 1, 'CPU fallback is disabled'),
])
def test_backend_fails_closed_before_loading_models(monkeypatch, backend, hip, platform, gpu, count, error):
    torch = SimpleNamespace(version=SimpleNamespace(hip=hip), cuda=SimpleNamespace(
        is_available=lambda: count > 0, device_count=lambda: count))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setattr('mtu_engine.backend.sys', SimpleNamespace(platform=platform))
    if error:
        with pytest.raises(RuntimeError, match=error):
            validate_backend(backend, gpu)
    else:
        validate_backend(backend, gpu)


def test_backend_dependency_locks_do_not_mix_gpu_runtimes():
    root = Path(__file__).resolve().parents[1]
    cuda = tomllib.loads((root / 'uv.lock').read_text(encoding='utf-8'))
    amd = tomllib.loads((root / 'environments/amd/uv.lock').read_text(encoding='utf-8'))
    cuda_names = {p['name'] for p in cuda['package']}
    amd_names = {p['name'] for p in amd['package']}
    assert 'onnxruntime-gpu' in cuda_names and 'onnxruntime-directml' not in cuda_names
    assert not any(n.startswith(('rocm', 'amd-torch')) for n in cuda_names)
    assert 'onnxruntime-directml' in amd_names
    assert not amd_names.intersection({'onnxruntime', 'onnxruntime-gpu'})
    assert not any(n.startswith('nvidia-') for n in amd_names)
    assert {'amd-torch-device-gfx1030', 'rocm-sdk-device-gfx1030'} <= amd_names
    assert ENGINE_DEFAULTS['backend'] == 'cuda'
