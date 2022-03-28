from typing import Dict
import tvm

_current_scope = None

class Scope(Dict):
    def __init__(self, schedule):
        self.schedule = schedule
        self.bounds = tvm.te.schedule.InferBound(self.schedule)
        self.shared_mem_outputs = []
        self._build_analyzer()
        self._get_grid_block_size()

    def _build_analyzer(self):
        self.analyzer = tvm.arith.Analyzer()
        for iterator, region in self.bounds.items():
            if isinstance(region.min, tvm.tir.expr.IntImm) and isinstance(region.extent, tvm.tir.expr.IntImm):
                if iterator.var.name.startswith("blockIdx"):
                    bound = tvm.arith.ConstIntBound(0, 0)
                else:
                    bound = tvm.arith.ConstIntBound(int(region.min), int(region.min) + int(region.extent) - 1)
                self.analyzer.update(iterator.var, bound)

    def _get_grid_block_size(self):
        grid_block_size = {
            "threadIdx.x" : 1, "threadIdx.y" : 1, "threadIdx.z" : 1,
            "blockIdx.x" : 1, "blockIdx.y" : 1, "blockIdx.z" : 1,
        }
        for iter_var, region in self.bounds.items():
            name = iter_var.var.name
            if name in grid_block_size:
                grid_block_size[name] = max(int(region.extent), grid_block_size[name])
        self.block_size = [grid_block_size[x] for x in ["threadIdx.x", "threadIdx.y", "threadIdx.z"]]
        self.grid_size = [grid_block_size[x] for x in ["blockIdx.x", "blockIdx.y", "blockIdx.z"]]
        print(grid_block_size)

    def __enter__(self):
        global _current_scope
        _current_scope = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _current_scope
        _current_scope = None

def get_scope():
    return _current_scope
