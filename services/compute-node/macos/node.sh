#!/bin/bash
set -euo pipefail
umask 077
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "需要原生 Apple Silicon macOS 14 或更新版本。" >&2
  exit 1
fi
root="$(cd -- "$(dirname -- "$0")" && pwd)"
if [[ ! -x "$root/runtime/bin/python3.12" ]]; then
  mkdir -p "$root/.setup/downloads"
  archive="$root/.setup/downloads/python.tar.gz"
  echo "[1/3] 下载并校验独立 Python..."
  curl --fail --location --retry 3 --progress-bar --output "$archive.partial" '@PYTHON_URL@'
  echo "@PYTHON_SHA256@  $archive.partial" | shasum -a 256 -c -
  mv "$archive.partial" "$archive"
  stage="$(mktemp -d "$root/.setup/python.XXXXXX")"
  tar -xzf "$archive" -C "$stage"
  mv "$stage/python" "$root/runtime"
  rmdir "$stage"
fi
if [[ "${1:-start}" != run && -f "$root/.setup/update/transaction.json" ]]; then
  "$root/runtime/bin/python3.12" -I -B -c \
    'import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1] + "/.setup/update/recovery"); from installer import recover; recover(Path(sys.argv[1]))' "$root"
fi
exec "$root/runtime/bin/python3.12" -I -B "$root/source/compute-node/macos/launcher.py" "${@:-start}"
