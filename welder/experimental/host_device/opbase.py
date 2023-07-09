import torch
import numpy as np
from pipeline import PipelineGlobal

class OpBase():
    def __init__(self, device) -> None:
        self.device = device
        self.pipeline = PipelineGlobal.get_pipeline()
        self.h2d_stream = PipelineGlobal.get_streams(device, "h2d")
        self.d2h_stream = PipelineGlobal.get_streams(device, "d2h")
        self.kernel_stream = PipelineGlobal.get_streams(device, "kernel")
        self.run_shape = [1]

    def h2d(self, args):
        results = []
        torch.cuda.set_stream(self.h2d_stream)
        for tensor in args[1]:
            results.append(tensor.to(self.device, non_blocking=True))
        self.h2d_stream.synchronize()
        return results

    def d2h(self, args):
        results = []
        torch.cuda.set_stream(self.d2h_stream)
        for tensor in args[1]:
            output_buffer = torch.empty(size=tensor.shape, pin_memory=True).copy_(tensor, non_blocking=True)
            results.append(output_buffer)
        self.d2h_stream.synchronize()
        return results

    def compute(self, args):
        raise NotImplementedError

    def slice(self, args):
        n = args[0]
        deps = self.get_dependency(n)
        results = []
        for tid, dep in enumerate(deps):
            deps_processed = []
            for idx, ran in enumerate(dep):
                if ran is None:
                    st, ed =  None, None
                else:
                    st, ed = max(0, ran[0]), min(ran[1], self.inputs[tid].shape[idx])
                deps_processed.append(slice(st, ed))
            sliced_tensor = self.inputs[tid].__getitem__(deps_processed)
            pinned_tensor = torch.empty(size=sliced_tensor.shape, pin_memory=True).copy_(sliced_tensor)
            results.append(pinned_tensor)
        return results

    def get_dependency(self, n):
        raise NotImplementedError

    def gather(self, args):
        n = args[0]
        t_index = self.get_tile_index(n)
        for tid, tensor in enumerate(args[1]):
            t_tile = self.tiles[tid]
            slices = [slice(t_index[i] * t_tile[i], min((t_index[i] + 1) * t_tile[i], self.outputs[tid].shape[i]))
                for i in range(len(tensor.shape))]
            self.outputs[tid].__getitem__(slices).copy_(tensor)

    def get_tile_index(self, n):
        result = []
        for shape in reversed(self.run_shape):
            result.append(n % shape)
            n //= shape
        return tuple(reversed(result))

    def run_pipeline(self):
        self.run_shape = [int(np.ceil(y / x)) for x, y in zip(self.tiles[0], self.outputs[0].shape)]
        self.pipeline.set_funcs([self.slice, self.h2d, self.compute, self.d2h, self.gather])
        self.pipeline.run(np.prod(self.run_shape))
