"""Cached online preparation and rollback updates; private data stays in place."""
import hashlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_support import download, safe_extract_zip, sha256
from run_node import checked_path

COMMANDS = {'Start.command': 'start', 'Stop.command': 'stop', 'Status.command': 'status',
            'Update.command': 'update', 'Configure.command': 'configure', 'Check.command': 'check',
            'Enable-Autostart.command': 'enable', 'Disable-Autostart.command': 'disable'}
PROGRAM = ('node.sh', *COMMANDS, 'source', 'packages', 'docs', 'README.md', 'LICENSE',
           'node.example.json', 'portable.json', 'runtime-requirements.txt', 'build-requirements.txt')
ASSETS = ('upstream', 'models', 'fonts', 'licenses', 'hyphenation', 'mtu-assets.json')
RELEASE_URL = 'https://api.github.com/repos/Wy2926/node-comics-compute/releases/latest'
ARCHIVE = 'NodeComicsNode-macos-arm64.zip'


def clone(source, target):
    if sys.platform == 'darwin':
        result = subprocess.run(['cp', '-cR', str(source), str(target)], capture_output=True)
        if result.returncode == 0:
            return
    shutil.copytree(source, target, dirs_exist_ok=True)


def validate(root):
    manifest = json.loads((root / 'portable.json').read_text())
    if manifest.get('platform') != 'macos-arm64' or manifest.get('format') != 'macos-portable-v1':
        raise ValueError('请选择 Apple Silicon Mac 部署包')
    for name, expected in manifest['files'].items():
        if name.split('/')[0] not in PROGRAM or sha256(checked_path(root, name)) != expected:
            raise ValueError('部署包文件或摘要不匹配：' + name)
    for name in ('node.sh', *COMMANDS):
        (root / name).chmod(0o755)
    return manifest


def prepare(root, previous=None):
    manifest = validate(root)
    cache = (previous or root) / '.setup/downloads'
    toolchain = manifest['toolchain']
    requirements = ('runtime-requirements.txt', 'build-requirements.txt')
    identity = hashlib.sha256((''.join(sha256(root / p) for p in requirements) +
                               json.dumps(toolchain, sort_keys=True)).encode()).hexdigest()
    runtime = root / 'runtime'
    marker = runtime / '.node-requirements'
    old_marker = previous / 'runtime/.node-requirements' if previous else marker
    reusable = old_marker.is_file() and old_marker.read_text() == identity
    python = runtime / 'bin/python3.12'
    if previous and reusable:
        python = previous / 'runtime/bin/python3.12'
    elif not marker.is_file() or marker.read_text() != identity:
        print('[2/3] 安装锁定依赖（复用下载缓存）...', flush=True)
        if not runtime.exists():
            old = json.loads((previous / 'portable.json').read_text()) if previous else {}
            if previous and old.get('toolchain', {}).get('python', {}).get('sha256') == toolchain['python']['sha256']:
                clone(previous / 'runtime', runtime)
            else:
                spec = toolchain['python']
                archive = download(spec['url'], spec['sha256'], cache / spec['sha256'])
                target = root / '.setup/python'
                target.mkdir(parents=True, exist_ok=True)
                with tarfile.open(archive) as package:
                    package.extractall(target, filter='data')
                (target / 'python').rename(runtime)
        spec = toolchain['uv']
        archive = download(spec['url'], spec['sha256'], cache / spec['sha256'])
        uv = cache / ('uv-' + spec['version'])
        with tarfile.open(archive) as package, package.extractfile('uv-aarch64-apple-darwin/uv') as source:
            with uv.open('wb') as target:
                shutil.copyfileobj(source, target)
        uv.chmod(0o755)
        env = dict(os.environ, UV_CACHE_DIR=str(cache / 'uv-cache'), UV_NO_CONFIG='1', UV_LINK_MODE='copy')
        for name in reversed(requirements):
            subprocess.run([str(uv), 'pip', 'install', '--python', str(python), '--require-hashes',
                            '--no-build-isolation', '-r', str(root / name)], env=env, check=True)
        marker.write_text(identity)
    current_lock = sha256(root / 'source/classic-engine/mtu_engine/upstream.lock.json')
    old_assets = previous / 'mtu-assets.json' if previous else root / 'mtu-assets.json'
    assets_match = old_assets.is_file() and json.loads(old_assets.read_text())['lock_sha256'] == current_lock
    if not (root / 'mtu-assets.json').exists() and not (previous and assets_match):
        print('[3/3] 准备并校验模型、字体和上游源码...', flush=True)
        stage = root / '.setup/assets'
        subprocess.run([str(python), '-I', '-B', str(root / 'source/classic-engine/tools/prepare_mtu.py'),
                        '--output', str(stage), '--cache', str(cache)], check=True)
        if previous:
            clone(previous / 'models', root / 'models')
        for name in ASSETS:
            if name == 'models' and (root / name).exists():
                shutil.copytree(stage / name, root / name, dirs_exist_ok=True)
            else:
                remove(root / name)
                (stage / name).rename(root / name)
    assets = root if (root / 'mtu-assets.json').exists() else previous
    code = ('import sys; sys.path.insert(0, sys.argv[1]); from mtu_engine.assets import verify; '
            'verify(sys.argv[2]); print("资产校验通过。")')
    subprocess.run([str(python), '-I', '-B', '-c', code, str(root / 'source/classic-engine'),
                    str(assets / 'models')], check=True)


def remove(path):
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def rollback(root, record):
    backup = root / '.setup/update/previous'
    stage = root / '.setup/update/candidate'
    for name, existed in reversed(list(record['entries'].items())):
        if (backup / name).is_file():
            (backup / name).replace(root / name)
        elif (backup / name).exists():
            remove(root / name)
            (backup / name).rename(root / name)
        elif not existed and not (stage / name).exists():
            remove(root / name)


def recover(root):
    import launcher
    journal = root / '.setup/update/transaction.json'
    if not journal.exists():
        return
    record = json.loads(journal.read_text())
    # Recovery also runs before importing source that an interrupted rename may have moved.
    with (root / '.setup/control.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launcher.stop(root)
        rollback(root, record)
        journal.unlink()
    if record['running']:
        launcher.start(root)
    print('已恢复中断升级前的版本，可再次更新。', flush=True)


def install_update(root, stage):
    import launcher
    prepare(stage, previous=root)
    work = root / '.setup/update'
    backup = work / 'previous'
    status = launcher.status(root)
    running = launcher.loaded(root) or (status['fresh'] and status.get('phase') == 'running')
    launcher.stop(root)
    switched = False
    journal = work / 'transaction.json'
    try:
        # Use the existing recovery check; unknown deliveries must keep their original engine.
        subprocess.run([str(root / 'node.sh'), 'pending'], check=True)
        remove(backup)
        backup.mkdir(parents=True)
        recovery = work / 'recovery'
        recovery.mkdir(exist_ok=True)
        for filename in ('installer.py', 'launcher.py'):
            shutil.copy2(stage / 'source/compute-node/macos' / filename, recovery / filename)
        for filename in ('build_support.py', 'run_node.py'):
            shutil.copy2(stage / 'source/compute-node' / filename, recovery / filename)
        entries = {name: (root / name).exists() for name in (*PROGRAM, *ASSETS, 'runtime')
                   if (stage / name).exists()}
        record = {'running': running, 'entries': entries}
        temporary = journal.with_suffix('.tmp')
        temporary.write_text(json.dumps(record))
        temporary.replace(journal)
        switched = True
        with (root / 'data/launcher.lock').open('a') as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for name, existed in entries.items():
                if (stage / name).is_file():
                    if existed:
                        saved = backup / (name + '.tmp')
                        shutil.copy2(root / name, saved)
                        saved.replace(backup / name)
                    (stage / name).replace(root / name)
                else:
                    if existed:
                        (root / name).rename(backup / name)
                    (stage / name).rename(root / name)
        # The transaction is still open; invoke the installed launcher directly, not bootstrap recovery.
        subprocess.run([str(root / 'runtime/bin/python3.12'), '-I', '-B',
                        str(root / 'source/compute-node/macos/launcher.py'), 'check'], check=True)
        if running:
            launcher.start(root)
        journal.unlink()
    except BaseException:
        if switched:
            launcher.stop(root)
            rollback(root, record)
            journal.unlink(missing_ok=True)
        if running:
            launcher.start(root)
        raise
    print('更新完成，旧版保存在 ' + str(backup), flush=True)


def update(root, archive=None):
    work = root / '.setup/update'
    work.mkdir(parents=True, exist_ok=True)
    if archive is None:
        print('检查最新稳定 Release...', flush=True)
        request = urllib.request.Request(RELEASE_URL, headers={'User-Agent': 'NodeComics-Mac/1'})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                release = json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            raise RuntimeError('尚无可访问的稳定 Release，请使用本地 ZIP 更新。') from None
        assets = {item['name']: item['browser_download_url'] for item in release['assets']}
        if ARCHIVE not in assets or ARCHIVE + '.sha256' not in assets:
            raise RuntimeError('最新 Release 尚未提供 Mac 部署包；也可运行 ./node.sh update /path/to/' + ARCHIVE)
        current = json.loads((root / 'portable.json').read_text())
        if current['version'].lstrip('v') == release['tag_name'].lstrip('v'):
            print('已是最新版。')
            return
        with urllib.request.urlopen(assets[ARCHIVE + '.sha256'], timeout=30) as response:
            checksum = response.read(1024).decode().split()[0]
        archive = download(assets[ARCHIVE], checksum, work / (checksum + '.zip'))
    else:
        archive = archive.resolve()
        checksum = archive.with_suffix('.zip.sha256').read_text().split()[0]
        if sha256(archive) != checksum:
            raise ValueError('更新 ZIP 摘要不匹配')
    unpack = work / 'unpack'
    remove(unpack)
    safe_extract_zip(archive, unpack)
    folders = list(unpack.iterdir())
    if len(folders) != 1 or not folders[0].is_dir():
        raise ValueError('更新 ZIP 必须包含一个节点目录')
    stage = work / 'candidate'
    remove(stage)
    folders[0].rename(stage)
    install_update(root, stage)
