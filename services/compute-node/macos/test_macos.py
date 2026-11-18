"""Deployment checks: preserve private state, recover switches, reuse prepared assets."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

pytest.importorskip('fcntl')
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import installer
import launcher


def write(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def package(root, version):
    for name in ('node.sh', 'source/compute-node/macos/installer.py',
                 'source/compute-node/macos/launcher.py', 'source/compute-node/build_support.py',
                 'source/compute-node/run_node.py', 'source/classic-engine/classic_node/__main__.py'):
        write(root, name, version)


@pytest.mark.parametrize('failure', [None, 'pending', 'backup', 'check'])
def test_update_preserves_identity_and_recovery_and_rolls_back_failed_checks(tmp_path, monkeypatch, failure):
    root = tmp_path / '节点 with spaces'
    stage = root / '.setup/update/candidate'
    package(root, 'old')
    package(stage, 'new')
    write(root, 'source/retired.py', 'retired')
    preserved = [write(root, name, 'private') for name in
                 ('data/node.json', 'data/state/journal.sqlite3', 'models/hayai/model.bin')]
    calls = []
    monkeypatch.setattr(installer, 'prepare', lambda *a, **kw: None)
    monkeypatch.setattr(launcher, 'status', lambda root: {'fresh': True, 'phase': 'running'})
    monkeypatch.setattr(launcher, 'loaded', lambda root: True)
    monkeypatch.setattr(launcher, 'stop', lambda root: calls.append('stop'))
    monkeypatch.setattr(launcher, 'start', lambda root: calls.append('start'))

    def run(command, **kwargs):
        operation = command[-1]
        calls.append(operation)
        if operation == failure:
            raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(installer.subprocess, 'run', run)
    copy = installer.shutil.copy2
    def copy_backup(source, target):
        if failure == 'backup' and target.parent.name == 'previous':
            target.write_text('incomplete backup')
            raise OSError('copy interrupted')
        return copy(source, target)
    monkeypatch.setattr(installer.shutil, 'copy2', copy_backup)
    if failure:
        with pytest.raises(OSError if failure == 'backup' else subprocess.CalledProcessError):
            installer.install_update(root, stage)
    else:
        installer.install_update(root, stage)
    expected = 'old' if failure else 'new'
    assert (root / 'node.sh').read_text() == expected
    assert (root / 'source/compute-node/macos/launcher.py').read_text() == expected
    assert (root / 'source/retired.py').exists() == bool(failure)
    assert all(path.read_text() == 'private' for path in preserved)
    assert not (root / '.setup/update/transaction.json').exists()
    assert calls[-1] == 'start'
    if failure == 'pending':
        assert calls == ['stop', 'pending', 'start']
    if not failure:
        assert (root / '.setup/update/previous/source/retired.py').read_text() == 'retired'


def test_recovery_handles_interruption_between_directory_renames(tmp_path, monkeypatch):
    root = tmp_path / 'node'
    backup = root / '.setup/update/previous'
    package(root, 'new')
    package(backup, 'old')
    installer.remove(root / 'source')
    write(root, 'Status.command', 'new entry')
    config = write(root, 'data/node.json', 'private identity')
    journal = write(root, '.setup/update/transaction.json', json.dumps({
        'running': True, 'entries': {'node.sh': True, 'source': True, 'Status.command': False}}))
    calls = []
    monkeypatch.setattr(launcher, 'stop', lambda root: calls.append('stop'))
    monkeypatch.setattr(launcher, 'start', lambda root: calls.append('start'))
    installer.recover(root)
    assert (root / 'node.sh').read_text() == 'old'
    assert (root / 'source/compute-node/macos/launcher.py').read_text() == 'old'
    assert not (root / 'Status.command').exists()
    assert config.read_text() == 'private identity' and not journal.exists()
    assert calls == ['stop', 'start']


def test_prepare_resumes_assets_then_reuses_runtime_and_models(tmp_path, monkeypatch):
    previous, stage = tmp_path / 'old', tmp_path / 'new'
    manifest = {'toolchain': {'python': {'sha256': 'p'}, 'uv': {'sha256': 'u'}}}
    monkeypatch.setattr(installer, 'validate', lambda _: manifest)
    for name in ('runtime-requirements.txt', 'build-requirements.txt'):
        write(stage, name, name)
    identity = hashlib.sha256((''.join(installer.sha256(stage / name) for name in
        ('runtime-requirements.txt', 'build-requirements.txt')) +
        json.dumps(manifest['toolchain'], sort_keys=True)).encode()).hexdigest()
    write(previous, 'runtime/.node-requirements', identity)
    lock = write(stage, 'source/classic-engine/mtu_engine/upstream.lock.json', '{}')
    for name in ('runtime-requirements.txt', 'build-requirements.txt',
                 'source/classic-engine/mtu_engine/upstream.lock.json'):
        write(previous, name, (stage / name).read_text())
    write(previous, 'upstream/partial', 'interrupted')
    extra = write(previous, 'models/hayai/private', 'keep')
    config = write(previous, 'data/node.json', 'identity')
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        if command[0] == 'cp':
            return subprocess.CompletedProcess(command, 1)  # Exercise the ordinary copy fallback on macOS.
        if '--output' in command:
            assets = Path(command[command.index('--output') + 1])
            for name in installer.ASSETS[:-1]:
                write(assets, name + '/prepared', '{}')
            write(assets, 'mtu-assets.json', json.dumps({'lock_sha256': installer.sha256(lock), 'files': {}}))
    monkeypatch.setattr(installer.subprocess, 'run', run)
    monkeypatch.setattr(installer, 'download', lambda *a: pytest.fail('downloaded unchanged environment'))
    installer.prepare(previous)
    assert not (previous / 'upstream/partial').exists()
    assert (previous / 'upstream/prepared').exists() and (previous / 'models/prepared').exists()
    commands.clear()
    installer.prepare(stage, previous)
    assert len(commands) == 1 and commands[0][0] == str(previous / 'runtime/bin/python3.12')
    assert commands[0][-1] == str(previous / 'models')
    assert extra.read_text() == 'keep' and config.read_text() == 'identity'
    assert not (stage / 'runtime').exists() and not (stage / 'models').exists()

    retired = write(previous, 'models/manga/pytorch_model.bin', 'retired')
    changed = write(previous, 'models/ocr/old-custom.onnx', 'user-modified')
    write(previous, 'mtu-assets.json', json.dumps({'lock_sha256': installer.sha256(lock), 'files': {
        'models/manga/pytorch_model.bin': installer.sha256(retired), 'models/ocr/old-custom.onnx': 'old-hash'}}))
    lock.write_text('new model inventory')
    installer.prepare(stage, previous)
    assert not (stage / 'models/manga/pytorch_model.bin').exists() and retired.is_file()
    assert (stage / 'models/ocr/old-custom.onnx').read_bytes() == changed.read_bytes()
    assert (stage / 'models/hayai/private').read_bytes() == extra.read_bytes()


def test_update_manifest_cannot_replace_private_data(tmp_path):
    write(tmp_path, 'portable.json', json.dumps({'format': 'macos-portable-v1',
          'platform': 'macos-arm64', 'files': {'data/node.json': '0' * 64}}))
    with pytest.raises(ValueError):
        installer.validate(tmp_path)


def test_stale_status_does_not_claim_center_connection(tmp_path, monkeypatch):
    import os
    directory = tmp_path / 'state'
    path = write(directory, 'status.json', json.dumps({'phase': 'running', 'connected': True}))
    monkeypatch.setattr(launcher, 'state', lambda root: directory)
    assert launcher.status(tmp_path)['connected']
    os.utime(path, (time.time() - 60,) * 2)
    assert not launcher.status(tmp_path)['connected']


def test_double_click_launcher_imports_adjacent_installer_in_isolated_python(tmp_path):
    script = write(tmp_path, 'source/compute-node/macos/launcher.py', Path(launcher.__file__).read_text())
    write(tmp_path, 'source/compute-node/macos/installer.py',
          'def prepare(root): print("prepared")\ndef update(*args): pass\n')
    write(tmp_path, 'node.example.json', '{"state_dir":"state"}')
    result = subprocess.run([sys.executable, '-I', '-B', str(script), 'prepare'],
                            check=True, capture_output=True, text=True)
    assert result.stdout == 'prepared\n'
    assert json.loads((tmp_path / 'data/node.json').read_text()) == {'state_dir': 'state'}
