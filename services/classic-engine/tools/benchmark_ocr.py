"""Measure the selected node OCR through its real analysis pipeline, offline."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import os
from pathlib import Path
import statistics
from time import perf_counter

import numpy as np
from PIL import Image

from classic_node.config import execution_settings, load
from classic_node.protocol import digest
from classic_node.runtime import Runtime
from classic_node.timing import collect
from mtu_engine.assets import file_hash
from mtu_engine.engine import serialize_region
from tools.validate_render_output import measured


def bounded_pages(pool, operation, paths, concurrency):
    """Keep at most one decoded page per worker and no executor backlog."""
    pending, results = {}, [None] * len(paths)
    remaining = iter(enumerate(paths))
    def submit():
        item = next(remaining, None)
        if item is not None:
            index, path = item
            pending[pool.submit(operation, path)] = index
    for _ in range(concurrency):
        submit()
    while pending:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            results[pending.pop(future)] = future.result()
            submit()
    return results


def summarize(pages):
    names = set().union(*(page['timings'] for page in pages))
    return {name: {'mean': statistics.mean(values), 'p50': statistics.median(values),
                   'p95': float(np.percentile(values, 95))}
            for name in sorted(names)
            if (values := [page['timings'].get(name, 0.) for page in pages])}

def run(args):
    if args.output.exists():
        raise ValueError('Use a new output file; do not overwrite an earlier comparison')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    config = load(args.config)
    if args.batch_size is not None:
        limit = 8 if config['engine']['ocr_backend'] == 'hayai' else 32
        if not config['engine'].get('ocr_min_batch_size', 1) <= args.batch_size <= limit:
            raise ValueError('batch-size is outside the selected OCR backend limits')
        config['engine']['ocr_batch_size'] = args.batch_size
    # Estimate the same page working buffers as the node before creating models.
    largest = 0
    for path in args.input:
        with Image.open(path) as image:
            largest = max(largest, image.width * image.height * 22)
    if largest * max(args.concurrency) > config['resident_bytes']:
        raise ValueError('Requested concurrency exceeds the configured page memory budget')
    # This tool performs analysis only; do not create unused render processes.
    config['render_workers'] = 1
    started = perf_counter()
    runtime = Runtime(config)
    import torch
    gpu = config['engine'].get('gpu', 0)
    accelerator = config['engine'].get('backend', 'cuda') != 'mps'
    report = {'version': runtime.version, 'ocr': runtime.engine.ocr.identity,
              'settings': execution_settings(config), 'startup_seconds': perf_counter() - started,
              'hardware': {'torch': torch.__version__, 'hip': torch.version.hip, 'cuda': torch.version.cuda,
                           'device': torch.cuda.get_device_name(gpu) if accelerator else 'mps'},
              'scope': 'Offline analysis; warmed shapes; no center or text-provider requests', 'runs': []}
    expected = {}

    def page(path):
        started = perf_counter()
        with Image.open(path) as image:
            rgb = np.array(image.convert('RGB'))
        decode_seconds = perf_counter() - started
        started = perf_counter()
        with collect() as timings:
            regions, *_ = runtime.engine.analyze(rgb)
        timings['analyze'] = perf_counter() - started
        timings['input_decode'] = decode_seconds
        timings['ocr_total'] = timings.get('analyze_route', 0) + timings.get('analyze_ocr', 0)
        texts = [region.text for region in regions]
        result = {'file': path.name, 'sha256': file_hash(path), 'regions': len(regions),
                  'text_sha256': digest(texts), 'timings': timings}
        if args.include_text:
            result.update(texts=texts, region_data=[serialize_region(region) for region in regions])
        return result

    try:
        for path in args.input:
            warmed = page(path)
            expected[warmed['sha256']] = warmed['text_sha256']
        for concurrency in args.concurrency:
            with ThreadPoolExecutor(concurrency, thread_name_prefix='ocr-benchmark') as pool:
                for repeat in range(args.repeats):
                    if accelerator:
                        torch.cuda.reset_peak_memory_stats(gpu)
                    pages, resources = measured(lambda: bounded_pages(pool, page, args.input, concurrency))
                    if accelerator:
                        resources.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(gpu),
                                         gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(gpu))
                    resources['cpu_core_equivalents'] = (resources['parent_cpu_seconds'] +
                        resources['children_cpu_seconds']) / resources['wall_seconds']
                    same = all(p['text_sha256'] == expected[p['sha256']] for p in pages)
                    run = {'concurrency': concurrency, 'repeat': repeat, **resources,
                           'pages_per_second': len(pages) / resources['wall_seconds'],
                           'same_text': same, 'summary': summarize(pages), 'pages': pages}
                    report['runs'].append(run)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                    print(json.dumps({key: run[key] for key in
                        ('concurrency', 'repeat', 'wall_seconds', 'pages_per_second', 'same_text')}), flush=True)
                    if not same:
                        raise ValueError('OCR results changed; inspect the hash-only report')
    finally:
        runtime.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--concurrency', type=int, nargs='+', choices=range(1, 33), default=[1])
    parser.add_argument('--batch-size', type=int, help='Override engine.ocr_batch_size for this benchmark')
    parser.add_argument('--include-text', action='store_true', help='Include private OCR text and geometry in the report')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    run(args)
