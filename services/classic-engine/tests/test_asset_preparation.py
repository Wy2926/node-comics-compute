"""Offline asset inventory and generated-directory upgrade checks; no models."""
import json
from pathlib import Path
import zipfile

import pytest

from mtu_engine.assets import LOCK, verify
from tools import prepare_mtu


def test_prepare_removes_retired_generated_assets_but_preserves_samples_and_cache(tmp_path, monkeypatch):
    output, cache = tmp_path / 'prepared', tmp_path / 'download-cache'
    stale = ('models/retired/weights.bin', 'models/hayai/processor/preprocessor_config.json',
             'upstream/retired/module.py', 'upstream/retired/LICENSE', 'licenses/retired-model.txt',
             'fonts/RetiredFont.ttf', 'licenses/RetiredFont-OFL.txt',
             'models/ocr/ocr_ar_48px.ckpt', 'models/ocr/alphabet-all-v7.txt', 'upstream/comic_translate/LICENSE')
    for name in stale:
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'old generated asset')
    sample = output / 'samples/private-page.png'
    sample.parent.mkdir()
    sample.write_bytes(b'private user sample')
    cache.mkdir()
    cached_sample = cache / 'kept-cache-entry'
    cached_sample.write_bytes(b'existing cache content')

    host = tmp_path / 'build-host'
    host.mkdir()
    (host / 'assets.json').write_text(json.dumps({'fonts': [{
        'name': 'Test.ttf', 'url': 'font', 'sha256': 'font-digest',
        'notice': 'Test-LICENSE.txt', 'notice_url': 'font-license', 'notice_sha256': 'notice-digest',
    }]}), encoding='utf-8')
    monkeypatch.setattr(prepare_mtu, 'HOST', host)
    archives = {}
    for key, repository, modules in (
        ('source', 'manga-translator-ui', ('manga_translator',)),
        ('ballons', 'BallonsTranslator', ('ballontranslator', 'resources')),
    ):
        source = LOCK[key]
        prefix = repository + '-' + source['revision'] + '/'
        entries = {prefix + name + '/__init__.py': '' for name in modules}
        entries[prefix + 'LICENSE'] = 'retained source license'
        if 'patches' in source:
            entries.update({prefix + name: '# Upstream license retained\n'
                            + ''.join(patch['before'] for patch in patches) + '# Other source retained\n'
                            for name, patches in source['patches'].items()})
        archives[source['url']] = entries
    archives[LOCK['hyphenation']['url']] = {
        'pyphen/dictionaries/hyph_en_US.dic': 'UTF-8\n',
        'pyphen/dictionaries/README_en_US.txt': 'retained dictionary notice',
    }
    def download(url, checksum, target):
        target = Path(target)
        if url in archives:
            with zipfile.ZipFile(target, 'w') as package:
                for name, contents in archives[url].items():
                    package.writestr(name, contents)
        else:
            target.write_bytes(b'prepared test asset')
        return target

    monkeypatch.setattr(prepare_mtu, 'download', download)
    manifest = prepare_mtu.prepare(output, cache)
    assert verify(output / 'models') == manifest
    for key in ('source', 'ballons'):
        for relative, patches in LOCK[key]['patches'].items():
            expected = '# Upstream license retained\n' + ''.join(p['before'] for p in patches) + '# Other source retained\n'
            for patch in patches:
                expected = expected.replace(patch['before'], patch['after'])
            assert (output / 'upstream' / relative).read_text(encoding='utf-8') == expected
    with pytest.raises(ValueError, match='target changed'):
        prepare_mtu.apply_source_patches(output / 'upstream')
    assert all(not (output / name).exists() for name in stale)
    assert {path.name for path in (output / 'fonts').iterdir()} == {'Test.ttf'}
    assert sample.read_bytes() == b'private user sample'
    assert cached_sample.read_bytes() == b'existing cache content'
    for name in ('upstream/manga_ocr/LICENSE', 'upstream/mot/LICENSE',
                 'models/ocr/ch_PP-OCRv5_rec_mobile_infer.onnx', 'models/manga/pytorch_model.bin'):
        assert name not in manifest['files']
    assert json.loads((output / 'models/hayai/source.json').read_text()) == LOCK['hayai']
    for folder, spec in LOCK['hayai'].items():
        assert all(f'models/hayai/{folder}/{name}' in manifest['files'] for name in spec['files'])
    assert not any(name.startswith('samples/') for name in manifest['files'])
    image_files = verify(output / 'models', include_hayai=False)['files']
    assert set(manifest['files']) - set(image_files) == {p for p in manifest['files'] if p.startswith('models/hayai/')}
    (output / 'models/hayai/model/model.safetensors').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='asset checksum mismatch'):
        verify(output / 'models')


@pytest.mark.parametrize('marker', ['node.json', 'state'])
def test_prepare_rejects_node_state_before_removing_generated_files(tmp_path, marker):
    (tmp_path / marker).write_bytes(b'node state')
    original = tmp_path / 'models/keep.bin'
    original.parent.mkdir()
    original.write_bytes(b'untouched')
    with pytest.raises(ValueError, match='configured node state'):
        prepare_mtu.prepare(tmp_path, tmp_path / 'cache')
    assert original.read_bytes() == b'untouched'


@pytest.mark.parametrize('anchor_count', [0, 2])
def test_preparation_rejects_missing_or_ambiguous_anchor_without_partial_write(tmp_path, monkeypatch, anchor_count):
    relative = 'manga_translator/rendering/__init__.py'
    patches = LOCK['source']['patches'][relative]
    source = ''.join(patch['before'] for patch in patches[:-1]) + patches[-1]['before'] * anchor_count
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_text(source, encoding='utf-8')
    monkeypatch.setitem(LOCK['source'], 'patches', {relative: patches})
    with pytest.raises(ValueError, match='target changed'):
        prepare_mtu.apply_source_patches(tmp_path)
    assert target.read_text(encoding='utf-8') == source


def test_preparation_rejects_patch_outside_bundle(tmp_path, monkeypatch):
    monkeypatch.setitem(LOCK['source'], 'patches', {'../outside.py': []})
    with pytest.raises(ValueError, match='Unsafe upstream patch'):
        prepare_mtu.apply_source_patches(tmp_path)


def test_previous_asset_lock_is_rejected_before_model_loading(tmp_path):
    manifest = {'upstream': LOCK['source']['revision'], 'lock_sha256': 'previous-unpatched-lock'}
    (tmp_path / 'mtu-assets.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='do not match'):
        verify(tmp_path / 'models')
