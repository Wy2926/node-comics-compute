"""Pinned Hayai weights, inference code, processor, tokenizer and SigLIP2 vision config."""
import json
from pathlib import Path

from mtu_engine.assets import LOCK, file_hash

SOURCES = LOCK['hayai']


def verify_hayai(directory):
    directory = Path(directory).resolve()
    manifest = directory / 'source.json'
    if json.loads(manifest.read_text()) != SOURCES:
        raise ValueError('Expected the pinned Hayai artifact')
    for folder, source in SOURCES.items():
        for name, checksum in source['files'].items():
            if file_hash(directory / folder / name) != checksum:
                raise ValueError('Hayai artifact checksum mismatch')
    return {'backend': 'hayai', 'manifest_sha256': file_hash(manifest)}
