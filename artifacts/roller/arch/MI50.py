from .Arch import *

class MI50(Arch):
    # compute 7.0
    def __init__(self, para_opt = True):
        super().__init__()
        self.num_level = 2
        # DRAM: memory level 0
        # SMEM: memory level 1
        # REG: memory level 2
        # bandwidth in GBps
        self.bandwidth = [900, 8000]
        # compute throughput in GFLOPS
        self.peak_flops = 12000
        self.limit = []
        self.reg_cap = [20000, 70]
        self.smem_cap = [34000]
        self.compute_max_core = [60]

        self.warp_size = 64
        self.compute_sm_partition = [60, 4]
        self.smem_sm_partition = [60, 4]
        self.compute_block_schedule_way = ["warp"]
        self.smem_block_schedule_way = ["warp"]
        self.transaction_size = [64, 128]   # in bytes
        self.glbmem_sm_partition = [60, 1]  # 1: active blocks
        self.smem_bank_size=4
