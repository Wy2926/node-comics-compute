"""Bounded startup-only CUDA/HIP graphs for the unchanged FP32 LaMa generator."""
from threading import Lock

import torch

from classic_node.timing import record
from .backend import CUDA_GRAPH_MIN_FREE


@torch.inference_mode()
def cuda_lama_pixels(model, image, mask, device):
    """Pinned FP32 LaMa Fourier path: transfer bytes in both directions."""
    pixels = torch.from_numpy(image).to(device).permute(2, 0, 1).unsqueeze(0)
    # A device FP32 divisor matches CPU division for all 256 input values.
    pixels = pixels.div(pixels.new_full((), 255, dtype=torch.float32))
    mask = torch.from_numpy(mask[:, :, 0]).to(device).float()[None, None]
    # The pinned generator already zeroes masked inputs and blends binary masks.
    with torch.autocast(device_type='cuda', enabled=False):
        result = model(pixels, mask)
    return result[0].permute(1, 2, 0).mul(255).to(torch.uint8).contiguous().cpu().numpy()


def tensor_key(*tensors):
    return tuple((value.shape, value.stride(), value.dtype, value.device) for value in tensors)


class LamaGraphs(torch.nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        self.graphs = {}
        self.lock = Lock()
        self.finished = None
        self.prepared = False
        self.eval()

    @torch.inference_mode()
    def prepare(self):
        # Called only during Engine.warmup, before worker threads accept pages.
        # Larger/non-square inputs stay eager: at most eight fixed graph pools.
        if self.prepared:
            return
        self.prepared = True
        device = next(self.generator.parameters()).device
        # Leave room for real detector/OCR pages on smaller or busy NVIDIA cards.
        if device.type == 'cuda' and torch.version.cuda:
            if torch.cuda.mem_get_info(device)[0] < CUDA_GRAPH_MIN_FREE:
                return
            self.finished = torch.cuda.Event()
        stream = torch.cuda.Stream(device=device)
        caller = torch.cuda.current_stream(device)
        stream.wait_stream(caller)
        with torch.cuda.stream(stream):
            for side in range(64, 513, 64):
                # Match the upstream NumPy HWC -> FP32 GPU tensor layout exactly.
                image = torch.ones((side, side, 3), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
                mask = torch.zeros((1, 1, side, side), dtype=torch.float32, device=device)
                mask[:, :, side // 4:side * 3 // 4, side // 4:side * 3 // 4] = 1
                for _ in range(3):
                    self.generator(image, mask)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph, stream=stream, capture_error_mode='thread_local'):
                        output = self.generator(image, mask)
                except RuntimeError:
                    # Optional graph support must not hide a broken model/device.
                    self.graphs.clear()
                    self.generator(image, mask)
                    stream.synchronize()
                    return
                self.graphs[tensor_key(image, mask)] = (graph, image, mask, output)
        stream.synchronize()

    def forward(self, image, mask, rel_pos=None, direct=None):
        state = self.graphs.get(tensor_key(image, mask)) if rel_pos is None and direct is None else None
        if state is None or self.training:
            record('inpaint_eager_blocks', 1)
            return self.generator(image, mask, rel_pos, direct)
        graph, static_image, static_mask, output = state
        # Replay beside upstream's input preparation, not on a different stream.
        # Order static-buffer reuse after the owned clone, including across streams.
        with self.lock, torch.inference_mode():
            stream = torch.cuda.current_stream(image.device)
            if self.finished is not None:
                stream.wait_event(self.finished)
            try:
                static_image.copy_(image)
                static_mask.copy_(mask)
                graph.replay()
                result = output.clone()
            finally:
                if self.finished is None:
                    stream.synchronize()
                else:
                    self.finished.record(stream)
        record('inpaint_graph_blocks', 1)
        return result
