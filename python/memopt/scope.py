from typing import Dict
import tvm

_current_scope = None

class Scope(Dict):
    def __init__(self, schedule):
        self.schedule = schedule
        self.bounds = tvm.te.schedule.InferBound(self.schedule)
        self.shared_mem_outputs = []
        self._build_analyzer()

    def _build_analyzer(self):
        self.analyzer = tvm.arith.Analyzer()
        for iterator, region in self.bounds.items():
            if isinstance(region.min, tvm.tir.expr.IntImm) and isinstance(region.extent, tvm.tir.expr.IntImm):
                if iterator.var.name.startswith("blockIdx"):
                    bound = tvm.arith.ConstIntBound(0, 0)
                else:
                    bound = tvm.arith.ConstIntBound(int(region.min), int(region.min) + int(region.extent) - 1)
                self.analyzer.update(iterator.var, bound)

    def __enter__(self):
        global _current_scope
        _current_scope = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _current_scope
        _current_scope = None

def get_scope():
    return _current_scope
