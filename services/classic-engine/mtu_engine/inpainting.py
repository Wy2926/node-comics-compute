"""Bounded startup-only HIP graphs for the unchanged FP32 LaMa generator."""
from threading import Lock

import torch

from classic_node.timing import record


def tensor_key(*tensors):
    return tuple((value.shape, value.stride(), value.dtype, value.device) for value in tensors)


class LamaGraphs(torch.nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        self.graphs = {}
        self.lock = Lock()
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
        # Complete the owned clone before another caller can reuse static buffers.
        with self.lock, torch.inference_mode():
            static_image.copy_(image)
            static_mask.copy_(mask)
            graph.replay()
            result = output.clone()
            torch.cuda.current_stream(image.device).synchronize()
        record('inpaint_graph_blocks', 1)
        return result
