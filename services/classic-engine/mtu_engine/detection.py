"""Inference-only preparation for the pinned DBNet."""
import os

import cv2
import torch
from torch import nn
from torch.nn.utils import fuse_conv_bn_eval
from torchvision.models.resnet import BasicBlock, ResNet

from classic_node.timing import stage


def prepare_detector(model, device, precision):
    if precision == 'auto':
        # On the RX 6900 XT FP16 saves VRAM but changes two real-page readings.
        precision = 'fp16' if device == 'mps' else 'fp32'
    if precision == 'fp16':
        model = fuse_detector(model).half()
    elif precision != 'fp32':
        raise ValueError('detect_precision must be auto, fp32 or fp16')
    # Both DB postprocessing and its debug view read only shrink logits (channel 0).
    model.conv_db = model.conv_db.binarize
    return model


def matching_opencl_gpu(device):
    # OpenCV's default adapter does not follow CUDA/HIP device masks.
    masks = ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL')
    if device != 'cuda' or any(os.environ.get(k) for k in masks):
        return False
    try:
        if not cv2.ocl.useOpenCL():
            return False
        cl = cv2.ocl.Device_getDefault()
        name, amd = cl.name(), cl.isAMD()
        if cl.type() != cv2.ocl.Device_TYPE_GPU or not name:
            return False
    except cv2.error:
        # An unavailable OpenCL ICD must not prevent native CPU preprocessing.
        return False
    def matches(index):
        gpu = torch.cuda.get_device_properties(index)
        arch = (getattr(gpu, 'gcnArchName', '') or '').partition(':')[0]
        return gpu.name == name or (amd and arch == name)
    # Identically named adapters cannot be disambiguated by OpenCV's Python API.
    return [i for i in range(torch.cuda.device_count()) if matches(i)] == [torch.cuda.current_device()]


def filter_detector_image(image, opencl, pixel_budget):
    with stage('detect_filter'):
        # The native bilateral filter precedes resizing: bound full-size device buffers.
        if opencl and image.shape[0] * image.shape[1] <= pixel_budget:
            return cv2.bilateralFilter(cv2.UMat(image), 17, 80, 80).get()
        return cv2.bilateralFilter(image, 17, 80, 80)


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
