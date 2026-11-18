"""Reuse the frozen 48px model's XPOS factors across CUDA/HIP decoder layers."""
from functools import lru_cache, partial

import torch


def _factors(native, length, offset, downscale):
    # Rotating (1, 0) yields the native scaled (cos, sin), including prefix centering.
    basis = native.scale.new_zeros((1, length, native.head_dim))
    basis[..., ::2] = 1
    rotation = native(basis, offset=offset, downscale=downscale)
    return tuple(rotation[..., i::2].repeat_interleave(2, dim=-1) for i in (0, 1))


class _ColorPositions(torch.nn.Module):
    def __init__(self, native):
        super().__init__()
        self.native = native
        self.scale_base = native.scale_base
        self.factors = lru_cache(32)(partial(_factors, native))

    @property
    def scale(self):
        return self.native.scale

    def forward(self, x, offset=0, downscale=False):
        from manga_translator.ocr.xpos_relative_position import rotate_every_two
        length = x.shape[1]
        # At 80 FP32 channels, 32 entries retain at most 20 MiB, regardless of page count.
        if not 0 < length + offset <= 1024 or not length:
            return self.native(x, offset=offset, downscale=downscale)
        cos, sin = self.factors(length, offset, downscale)
        return x * cos + rotate_every_two(x) * sin


def prepare_color_positions(model):
    from manga_translator.ocr.xpos_relative_position import XPOS
    layers = [layer for layer in model.modules() if isinstance(getattr(layer, 'xpos', None), XPOS)]
    if not layers:
        return
    shared = _ColorPositions(layers[0].xpos).eval()
    for layer in layers:
        if layer.xpos.scale_base == shared.scale_base and torch.equal(layer.xpos.scale, shared.scale):
            layer.xpos = shared
