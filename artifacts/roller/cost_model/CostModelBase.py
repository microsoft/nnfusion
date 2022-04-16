from config import *
from op import *

class CostModelBase:
    def __init__(self):
        pass

    def compute_estimate(self, schedule, tile_tensor = "output"):
        # estimate the latency of compute
        raise NotImplementedError

    def memory_estimate(self, schedule, mem_level, tile_tensor = "output"):
        # estimate the latency of memory latency for a given memory level
        raise NotImplementedError

    def Theoretical_Perf(self, schedule, pure_memory = False, tile_tensor = "output"):
        mem_level = self.arch.num_level - 1
        if pure_memory:
            perf = 0
        else:
            perf = self.compute_estimate(schedule, tile_tensor)
        while mem_level >= 0:
            perf = max(perf, self.memory_estimate(schedule, mem_level, tile_tensor))
            mem_level -= 1
        return perf
