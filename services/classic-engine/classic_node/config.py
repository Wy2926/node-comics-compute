"""Local implementation settings; central configuration never sets GPU threads."""
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from .resources import resolve_cpu_resources

ENGINE_DEFAULTS = {'models': '.assets/models', 'gpu': 0, 'threads': 'auto', 'backend': 'cuda',
    'ocr_model': None,
    'font': [], 'detect_size': 1280, 'detect_precision': 'auto', 'inpainting_size': 512, 'keep_lang': None,
    'ocr_batch_size': 4, 'detect_batch_size': 4}
LOCAL_DEFAULTS = {'local_pages': 2, 'render_workers': 'auto', 'max_leases': 8,
    'download_workers': 4, 'delivery_workers': 4, 'claim_batch_size': 4,
    'document_control_workers': 2, 'document_require_sandbox': True,
    'resident_bytes': 1024 ** 3, 'document_spool_bytes': 6 * 1024 ** 3}


def execution_settings(config):
    """An explicit, credential/path-free allowlist for diagnostics and reports."""
    local = {key: config.get(key, default) for key, default in LOCAL_DEFAULTS.items()}
    local['inpaint_slots'] = config.get('inpaint_slots', local['local_pages'])
    local['analysis_workers'] = config.get('analysis_workers', local['download_workers'])
    local['engine'] = {key: value for key, value in (ENGINE_DEFAULTS | config.get('engine', {})).items()
                      if key in ENGINE_DEFAULTS and key not in {'models', 'font', 'keep_lang', 'ocr_model'}}
    local['engine']['ocr_backend'] = 'hayai'
    return local


def origin(value, allow_http=False):
    parsed = urlsplit(value)
    if (parsed.scheme not in (('https', 'http') if allow_http else ('https',)) or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('', '/')):
        raise ValueError('Expected an HTTPS origin without path, credentials or query')
    return value.rstrip('/')


def load(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if value.get('protocol_version') != 3:
        raise ValueError('protocol_version must be 3')
    if value.get('allow_http', False):
        raise ValueError('Production nodes require HTTPS')
    value['control_url'] = origin(value['control_url'])
    if value.get('proxy_url') is not None:
        import httpx
        try:
            if not httpx.Proxy(value['proxy_url']).url.host:
                raise ValueError
        except (TypeError, ValueError, httpx.InvalidURL):
            # Native validation errors can contain proxy credentials.
            raise ValueError('proxy_url must be an HTTP(S) or SOCKS5 proxy URL, or null') from None
    if value.get('control_ca'):
        value['control_ca'] = str((path.parent / value['control_ca']).resolve())
        if not Path(value['control_ca']).is_file():
            raise ValueError('control_ca must name an existing PEM trust bundle')
    value['node_token'] = os.environ.get('NODE_TOKEN') or value.get('node_token')
    if not value['node_token'] or not value['node_id'] or not value['resource_id']:
        raise ValueError('Node identity and credential are required')
    engine = value.get('engine', {})
    if engine.pop('ocr_backend', 'hayai') != 'hayai':
        raise ValueError('Only Hayai OCR is supported; remove engine.ocr_backend and prepare the new assets')
    # Previously shipped node.json files include these retired model settings.
    for key in ('ocr_min_batch_size', 'ocr_arena_extend_strategy', 'ocr_arena_shrinkage', 'color_batch_size'):
        engine.pop(key, None)
    if set(engine) - set(ENGINE_DEFAULTS):
        raise ValueError('Unknown local engine option')
    value['engine'] = ENGINE_DEFAULTS | engine
    for key, default in LOCAL_DEFAULTS.items():
        value.setdefault(key, default)
    value.setdefault('inpaint_slots', value['local_pages'])
    value.setdefault('analysis_workers', value['download_workers'])
    value['state_dir'] = str((path.parent / value.get('state_dir', 'state')).resolve())
    value['engine']['models'] = str((path.parent / value['engine']['models']).resolve())
    value['engine']['font'] = [str((path.parent / font).resolve()) for font in value['engine']['font']]
    if (not all(type(value[k]) is int for k in ('local_pages', 'max_leases'))
            or not 1 <= value['local_pages'] <= value['max_leases'] <= 32):
        raise ValueError('Require 1 <= local_pages <= max_leases <= 32')
    if type(value['inpaint_slots']) is not int or not 1 <= value['inpaint_slots'] <= value['local_pages']:
        raise ValueError('inpaint_slots must be between 1 and local_pages')
    if value['render_workers'] != 'auto' and (type(value['render_workers']) is not int
                                            or not 1 <= value['render_workers'] <= value['max_leases']):
        raise ValueError('render_workers must be auto or between 1 and max_leases')
    if not all(type(value[k]) is int and 1 <= value[k] <= 16
               for k in ('download_workers', 'analysis_workers', 'delivery_workers')):
        raise ValueError('Network worker counts must be between 1 and 16')
    if type(value['claim_batch_size']) is not int or not 1 <= value['claim_batch_size'] <= 4:
        raise ValueError('claim_batch_size must be between 1 and the center limit of 4')
    if type(value['document_control_workers']) is not int or not 2 <= value['document_control_workers'] <= 3:
        raise ValueError('document_control_workers must be 2 or 3 (heartbeat, sync)')
    if type(value['document_require_sandbox']) is not bool:
        raise ValueError('document_require_sandbox must be a boolean')
    if type(value['resident_bytes']) is not int or value['resident_bytes'] < 512 * 1024 * 1024:
        raise ValueError('Reserve at least 512 MiB for page buffers')
    if type(value['document_spool_bytes']) is not int or not 256 * 1024 ** 2 <= value['document_spool_bytes'] <= 16 * 1024 ** 3:
        raise ValueError('document_spool_bytes must be between 256 MiB and 16 GiB')
    if value['engine']['threads'] != 'auto' and (type(value['engine']['threads']) is not int or value['engine']['threads'] < 1):
        raise ValueError('engine.threads must be auto or a positive integer')
    if type(value['engine']['gpu']) is not int or value['engine']['gpu'] < 0:
        raise ValueError('gpu must be a nonnegative device index')
    from mtu_engine.backend import BACKENDS
    if value['engine']['backend'] not in BACKENDS:
        raise ValueError('backend must be cuda, amd or mps')
    if value['engine']['ocr_model'] is not None:
        if not isinstance(value['engine']['ocr_model'], str) or not value['engine']['ocr_model']:
            raise ValueError('engine.ocr_model must be a nonempty path or null')
        value['engine']['ocr_model'] = str((path.parent / value['engine']['ocr_model']).resolve())
    if type(value['engine']['inpainting_size']) is not int or value['engine']['inpainting_size'] not in (512, 768, 1024):
        raise ValueError('inpainting_size must be 512, 768 or 1024')
    if type(value['engine']['detect_size']) is not int or not 512 <= value['engine']['detect_size'] <= 2048:
        raise ValueError('detect_size must be between 512 and 2048')
    if value['engine']['detect_precision'] not in ('auto', 'fp32', 'fp16'):
        raise ValueError('detect_precision must be auto, fp32 or fp16')
    for key in ('ocr_batch_size', 'detect_batch_size'):
        limit = 8 if key == 'ocr_batch_size' else 32
        if type(value['engine'][key]) is not int or not 1 <= value['engine'][key] <= limit:
            raise ValueError(f'engine.{key} must be between 1 and {limit}')
    if value['engine']['keep_lang'] is not None:
        from langcodes import tag_is_valid
        language = value['engine']['keep_lang']
        if not isinstance(language, str) or len(language) != 2 or not language.islower() or not tag_is_valid(language):
            raise ValueError('keep_lang must be an ISO 639-1 source language code or null')
    resources = resolve_cpu_resources(analysis_threads=value['engine']['threads'], render_workers=value['render_workers'],
                                      local_pages=value['local_pages'], max_leases=value['max_leases'])
    value['engine']['threads'] = resources['analysis_threads']
    value['render_workers'] = resources['render_workers']
    value['_render_threads'] = resources['render_threads']
    value['_cpu_resources'] = resources
    return value
