"""Fold the pinned DBNet's eval normalization before low-precision inference."""
from torch import nn
from torch.nn.utils import fuse_conv_bn_eval
from torchvision.models.resnet import BasicBlock, ResNet


def fuse_detector(model):
    for parent in model.modules():
        pairs = []
        if isinstance(parent, nn.Sequential):
            children = list(parent.named_children())
            pairs = [(left, right) for (left, conv), (right, bn) in zip(children, children[1:])
                     if isinstance(conv, (nn.Conv2d, nn.ConvTranspose2d)) and isinstance(bn, nn.BatchNorm2d)]
        elif isinstance(parent, BasicBlock):
            pairs = [('conv1', 'bn1'), ('conv2', 'bn2')]
        elif isinstance(parent, ResNet):
            pairs = [('conv1', 'bn1')]
        for conv_name, bn_name in pairs:
            conv, bn = getattr(parent, conv_name), getattr(parent, bn_name)
            setattr(parent, conv_name, fuse_conv_bn_eval(conv, bn, transpose=isinstance(conv, nn.ConvTranspose2d)))
            setattr(parent, bn_name, nn.Identity())
    # Unfolded FP16 normalization changed recall on the real-page regression.
    if any(isinstance(module, nn.BatchNorm2d) for module in model.modules()):
        raise ValueError('Unsupported detector normalization')
    return model
