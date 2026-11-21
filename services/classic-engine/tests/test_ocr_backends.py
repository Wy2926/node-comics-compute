"""Hayai configuration, ordered paragraphs and pinned local assets."""
import asyncio
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from classic_node.config import execution_settings, load
from mtu_engine.ocr.crops import line_context_boxes
from mtu_engine.ocr.hayai import HayaiOCR, complete_texts


def test_public_ocr_and_color_imports_do_not_load_either_runtime():
    subprocess.run([sys.executable, '-c', '''
import sys
import mtu_engine.ocr
import mtu_engine.colors
assert not {'torch', 'transformers', 'rapidocr', 'manga_ocr', 'onnxruntime'} & sys.modules.keys()
'''], check=True)


def configuration(tmp_path, **options):
    path = tmp_path / 'node.json'
    path.write_text(json.dumps({'protocol_version': 3, 'control_url': 'https://example.test',
        'node_id': 'test', 'node_token': 'fixture', 'resource_id': 'test',
        'engine': {'backend': 'mps', 'ocr_backend': 'hayai', 'ocr_model': 'prepared', 'ocr_batch_size': 8} | options}))
    return path


def test_hayai_configuration_resolves_model_and_omits_path_in_diagnostics(tmp_path):
    result = load(configuration(tmp_path))
    assert result['engine']['ocr_model'] == str(tmp_path / 'prepared')
    public = execution_settings(result)['engine']
    assert public['ocr_backend'] == 'hayai' and 'ocr_model' not in public


@pytest.mark.parametrize('option', [
    {'ocr_backend': 'auto'}, {'ocr_model': ''},
    {'ocr_batch_size': True}, {'ocr_batch_size': '8'}, {'ocr_batch_size': 0}, {'ocr_batch_size': 9},
])
def test_invalid_backend_configuration_fails_before_model_load(tmp_path, option):
    with pytest.raises(ValueError):
        load(configuration(tmp_path, **option))


def test_direct_hayai_load_rejects_oversize_batches_before_loading_weights():
    with pytest.raises(ValueError, match='batch_size'):
        HayaiOCR('missing', batch_size=9)


def block(box, lines=1):
    return SimpleNamespace(xyxy=box, lines=[object()] * lines)


def test_paragraph_batches_are_bounded_and_restore_geometry_order():
    batches = []
    def recognize(crops):
        batches.append([crop.size for crop in crops])
        return [str(crop.size) for crop in crops]
    ocr = HayaiOCR.__new__(HayaiOCR)
    ocr.batch_size, ocr._recognize = 2, recognize
    regions = [block((0, 0, 20, 20), 3), block((-3, -3, 2, 2)),
               block((2, 2, 5, 6)), block((30, 30, 40, 40))]
    result = asyncio.run(ocr.recognize(np.zeros((20, 20, 3), np.uint8), regions, None))
    assert batches == [[(2, 2), (3, 4)], [(20, 20)]]
    assert [r.texts for r in result] == [['(20, 20)', '', ''], ['(2, 2)'], ['(3, 4)'], []]
    assert regions[0].xyxy == (0, 0, 20, 20) and len(regions[0].lines) == 3


def test_line_context_stays_on_page_and_outside_neighboring_paragraphs():
    from classic_node.protocol import MAX_REGIONS
    boxes = [(4, 3, 68, 15), (10, 60, 75, 76), (80, 69, 100, 80),
             (120, 20, 135, 75), (20, 120, 100, 135), (130, 175, 188, 197)]
    blocks = [block(box, 2 if i == 4 else 1) for i, box in enumerate(boxes)]
    crops = line_context_boxes(boxes, blocks, range(len(boxes)), 200, 200)
    assert crops == [(0, 0, 84, 31), *boxes[1:3], (112, 5, 143, 90), boxes[4], (114, 159, 200, 200)]
    assert boxes == [b.xyxy for b in blocks]
    crowded = [boxes[0]] * (MAX_REGIONS + 1)
    assert line_context_boxes(crowded, [blocks[0]] * len(crowded), range(len(crowded)), 200, 200) is crowded


def test_short_vertical_context_cannot_read_another_line_or_expand_a_long_caption():
    boxes = [(10, 10, 30, 66), (12, 72, 32, 128), (70, 5, 90, 150)]
    assert line_context_boxes(boxes, [block(b) for b in boxes], range(3), 100, 160) == boxes


def test_empty_paragraphs_cannot_authorize_erasure_and_count_mismatch_fails():
    ocr = HayaiOCR.__new__(HayaiOCR)
    ocr.batch_size, ocr._recognize = 8, lambda crops: ['', '']
    rows = asyncio.run(ocr.recognize(np.zeros((10, 10, 3), np.uint8),
        [block((0, 0, 5, 5)), block((0, 0, 5, 5), 2)], None))
    assert not any(''.join(row.texts) for row in rows)
    with pytest.raises(ValueError):
        asyncio.run(ocr.recognize(np.zeros((10, 10, 3), np.uint8), [block((0, 0, 5, 5))], None))


@pytest.mark.parametrize('indices', [[0], [1, 0]])
def test_processor_cannot_assign_text_to_the_wrong_crop(monkeypatch, indices):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace())
    ocr = HayaiOCR.__new__(HayaiOCR)
    ocr.processor = lambda **_: {'valid_indices': indices}
    with pytest.raises(ValueError, match='skipped or reordered'):
        ocr._recognize([Image.new('RGB', (8, 8))] * 2)


def test_explicit_retired_ocr_is_rejected_before_model_loading(tmp_path):
    with pytest.raises(ValueError, match='Only Hayai'):
        load(configuration(tmp_path, ocr_backend='classic'))


def test_truncated_or_empty_generation_cannot_authorize_erasure():
    tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0,
        decode=lambda tokens, **kw: ''.join({3: '日', 4: '本'}.get(t, '') for t in tokens))
    assert complete_texts([[1, 3, 4, 2, 0], [1, 3, 4, 3, 4], [1, 2, 0], [1, 3, 0]], tokenizer) == [
        '日本', '', '', '日']


@pytest.mark.parametrize('corrupt', [None, 'modeling_hayai.py', 'processing_hayai.py'])
def test_preparation_publishes_only_verified_self_contained_assets(tmp_path, monkeypatch, corrupt):
    from tools import prepare_hayai
    from mtu_engine.ocr import models
    source, output = tmp_path / 'source', tmp_path / 'prepared'
    sources = {}
    for folder, spec in models.SOURCES.items():
        (source / folder).mkdir(parents=True)
        files = {}
        for name in spec['files']:
            path = source / folder / name
            path.write_text('pinned ' + name)
            files[name] = models.file_hash(path)
        sources[folder] = {'repo': 'fixture/' + folder, 'revision': 'pinned', 'license': 'Apache-2.0', 'files': files}
    monkeypatch.setattr(models, 'SOURCES', sources)
    monkeypatch.setattr(prepare_hayai, 'SOURCES', sources)
    if corrupt:
        (source / 'model' / corrupt).write_text('modified executable code')
        with pytest.raises(ValueError, match='checksum'):
            prepare_hayai.prepare(source, output)
        assert not output.exists()
    else:
        assert prepare_hayai.prepare(source, output)['backend'] == 'hayai'
        assert prepare_hayai.prepare(tmp_path / 'absent', output) == models.verify_hayai(output)
        for folder, spec in sources.items():
            for name in spec['files']:
                assert (output / folder / name).read_bytes() == (source / folder / name).read_bytes()
        (output / 'model/model.safetensors').write_text('changed')
        with pytest.raises(ValueError, match='checksum'):
            models.verify_hayai(output)
    assert not list(tmp_path.glob('hayai-*'))
