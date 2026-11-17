"""Build the small Apple Silicon online-install ZIP; no models or private state."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

HOST = Path(__file__).resolve().parents[1]
REPO = HOST.parents[1]
sys.path.insert(0, str(HOST))
from build_support import archive_release, copy_sources, sha256
from macos.installer import COMMANDS


def build(output, version, uv):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    engine = REPO / 'services/classic-engine'
    for name in ('classic_node', 'mtu_engine', 'tools', 'docs', 'environments/macos'):
        copy_sources(engine / name, output / 'source/classic-engine' / name)
    for name in ('pyproject.toml', 'uv.lock', 'README.md', 'ENGINE.md', 'THIRD_PARTY.md', 'LICENSE'):
        shutil.copy2(engine / name, output / 'source/classic-engine' / name)
    for name in ('run_node.py', 'build_support.py', 'assets.json'):
        target = output / 'source/compute-node' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HOST / name, target)
    macos = output / 'source/compute-node/macos'
    macos.mkdir()
    for name in ('launcher.py', 'installer.py', 'toolchain.lock.json'):
        shutil.copy2(HOST / 'macos' / name, macos / name)
    copy_sources(REPO / 'packages/document-files', output / 'packages/document-files')
    copy_sources(REPO / 'docs', output / 'docs')
    shutil.copy2(REPO / 'LICENSE', output / 'LICENSE')
    shutil.copy2(HOST / 'macos/README.md', output / 'README.md')
    toolchain = json.loads((HOST / 'macos/toolchain.lock.json').read_text())
    script = (HOST / 'macos/node.sh').read_text()
    for key in ('url', 'sha256'):
        script = script.replace('@PYTHON_' + key.upper() + '@', toolchain['python'][key])
    (output / 'node.sh').write_text(script)
    for name, command in COMMANDS.items():
        (output / name).write_text(
            '#!/bin/bash\nroot="$(cd -- "$(dirname -- "$0")" && pwd)"\n'
            f'bash "$root/node.sh" {command} "$@"\nresult=$?\n'
            'echo\nread -r -p "按回车关闭窗口..." _\nexit "$result"\n')
    for path in [output / 'node.sh', *output.glob('*.command')]:
        path.chmod(0o755)
    config = json.loads((engine / 'node.example.json').read_text())
    config['resource_id'] = 'machine:mps:0'
    config['engine'].update(backend='mps', models='../models', detect_batch_size=1,
                            font=['../fonts/' + Path(name).name for name in config['engine']['font']])
    (output / 'node.example.json').write_text(json.dumps(config, indent=2) + '\n')
    for name, groups in (('runtime', ['--no-dev']), ('build', ['--only-group', 'build'])):
        subprocess.run([uv, 'export', '--project', str(engine / 'environments/macos'),
                        '--frozen', '--no-emit-project', '--no-emit-package', 'node-comics-classic-engine',
                        '--no-emit-package', 'comic-document-files', '--no-editable', '--no-header',
                        '--no-annotate', *groups, '--output-file', str(output / f'{name}-requirements.txt')],
                       check=True, stdout=subprocess.DEVNULL)
    manifest = {'format': 'macos-portable-v1', 'platform': 'macos-arm64', 'version': version,
                'toolchain': toolchain,
                'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                'source_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO)),
                'files': {p.relative_to(output).as_posix(): sha256(p)
                          for p in sorted(output.rglob('*')) if p.is_file()}}
    (output / 'portable.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return archive_release(output, 'portable.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--uv', default='uv')
    args = parser.parse_args()
    print(build(args.output, args.version, args.uv))
