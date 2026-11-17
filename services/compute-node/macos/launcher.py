"""Double-click controls and the foreground entry point supervised by launchd."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import plistlib
import runpy
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

# shortcut: one managed node per macOS user, use per-directory labels for multiple nodes.
LABEL = 'net.nodelane.comics.compute-node'
DOMAIN = f'gui/{os.getuid()}'
SERVICE = DOMAIN + '/' + LABEL
AUTOSTART = Path.home() / 'Library/LaunchAgents' / (LABEL + '.plist')


def environment(root):
    for key in list(os.environ):
        if key.startswith(('PYTHON', 'QT_', 'DYLD_')) or key in ('VIRTUAL_ENV', 'NODE_TOKEN'):
            os.environ.pop(key)
    os.environ.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      QT_QPA_PLATFORM='offscreen', PYTORCH_ENABLE_MPS_FALLBACK='0')
    for key, relative in {
        'TMPDIR': 'tmp', 'TMP': 'tmp', 'TEMP': 'tmp', 'XDG_CACHE_HOME': 'cache',
        'XDG_CONFIG_HOME': 'config', 'XDG_DATA_HOME': 'share', 'HF_HOME': 'cache/huggingface',
        'TORCH_HOME': 'cache/torch', 'MPLCONFIGDIR': 'cache/matplotlib', 'YOLO_CONFIG_DIR': 'cache/ultralytics',
    }.items():
        path = root / 'data' / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.chdir(root)


def initialize(root):
    config = root / 'data/node.json'
    config.parent.mkdir(parents=True, exist_ok=True)
    if not config.exists():
        with config.open('x', encoding='utf-8') as stream:
            stream.write((root / 'node.example.json').read_text())
        config.chmod(0o600)
    return config


def state(root):
    config = initialize(root)
    directory = Path(json.loads(config.read_text()).get('state_dir', 'state'))
    return (config.parent / directory).resolve()


def status(root):
    path = state(root) / 'status.json'
    try:
        value = json.loads(path.read_text())
        value['fresh'] = time.time() - path.stat().st_mtime < 10
        value['connected'] = bool(value.get('connected') and value['fresh'])
        return value
    except (OSError, ValueError):
        return {'phase': 'stopped', 'connected': False, 'fresh': False}


def loaded(root):
    result = subprocess.run(['launchctl', 'print', SERVICE], capture_output=True, text=True)
    if result.returncode:
        return False
    if str(root / 'node.sh') not in result.stdout:
        raise RuntimeError('当前用户已有其他目录的节点服务，请先停止该节点。')
    return True


def wait_ready(root):
    deadline = time.monotonic() + 600
    last = None
    while time.monotonic() < deadline:
        value = status(root)
        phase = value.get('phase')
        if phase != last:
            print('节点状态：' + str(phase), flush=True)
            last = phase
        if value['connected']:
            print('connected=true，中心心跳正常。', flush=True)
            return
        if value['fresh'] and phase == 'failed':
            break
        time.sleep(1)
    raise RuntimeError('节点未就绪，请查看 data/state/logs/node.log 和 data/logs/')


def start(root):
    (state(root) / 'stop').unlink(missing_ok=True)
    if loaded(root):
        value = status(root)
        if not value['fresh'] or value.get('phase') in ('stopped', 'failed'):
            subprocess.run(['launchctl', 'kickstart', SERVICE], check=True)
    else:
        logs = root / 'data/logs'
        logs.mkdir(parents=True, exist_ok=True)
        path = AUTOSTART if AUTOSTART.exists() else root / '.setup/launchd.plist'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(plistlib.dumps({
            'Label': LABEL, 'ProgramArguments': [str(root / 'node.sh'), 'run'],
            'WorkingDirectory': str(root), 'RunAtLoad': True,
            'KeepAlive': {'SuccessfulExit': False}, 'ExitTimeOut': 90, 'ThrottleInterval': 30,
            'ProcessType': 'Interactive', 'LimitLoadToSessionType': 'Aqua', 'Umask': 63,
            'StandardOutPath': str(logs / 'launchd.out.log'),
            'StandardErrorPath': str(logs / 'launchd.err.log'),
        }))
        subprocess.run(['launchctl', 'bootstrap', DOMAIN, str(path)], check=True)
    wait_ready(root)


def stop(root):
    active = loaded(root)
    directory = state(root)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'stop').touch()
    deadline = time.monotonic() + 120
    while True:
        with (root / 'data/launcher.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError('节点仍在排空，保留原程序与恢复数据后重试。')
        time.sleep(.5)
    if active:
        subprocess.run(['launchctl', 'bootout', SERVICE], check=True)
    print('节点已停止，恢复数据已保留。', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('start', 'stop', 'status', 'update', 'enable', 'disable',
                                           'configure', 'check', 'run', 'pending', 'prepare'))
    parser.add_argument('package', nargs='?', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    os.umask(0o077)
    config = initialize(root)
    if args.command == 'status':
        print(json.dumps(status(root), ensure_ascii=False, indent=2))
        return
    if args.command in ('run', 'check', 'pending'):
        environment(root)
        with (root / 'data/launcher.lock').open('a') as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            sys.path.insert(0, str(root / 'source/compute-node'))
            from run_node import activate_app, prepare_app
            activate_app(root, prepare_app(root))
            sys.argv = ['classic_node', args.command, '--config', str(config)]
            runpy.run_module('classic_node', run_name='__main__')
        return
    from installer import prepare, update
    (root / '.setup').mkdir(exist_ok=True)
    with (root / '.setup/control.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == 'configure':
            print('配置文件：' + str(config))
            subprocess.run(['open', '-t', str(config)], check=True)
        elif args.command == 'stop':
            stop(root)
        elif args.command == 'disable':
            AUTOSTART.unlink(missing_ok=True)
            print('已关闭登录自启；当前运行的节点继续运行。')
        elif args.command == 'update':
            update(root, args.package)
        else:
            prepare(root)
            if args.command == 'prepare':
                return
            value = json.loads(config.read_text())
            if value.get('control_url') == 'https://control.example.com' or any(
                    str(value.get(key, '')).startswith('REPLACE_') for key in ('node_id', 'node_token')):
                subprocess.run(['open', '-t', str(config)], check=True)
                print('请填写中心地址、节点 ID、令牌及资源 ID，保存后再次启动。')
                return
            if args.command == 'enable':
                AUTOSTART.parent.mkdir(parents=True, exist_ok=True)
                if loaded(root):
                    stop(root)
                AUTOSTART.touch()
            start(root)


if __name__ == '__main__':
    main()
