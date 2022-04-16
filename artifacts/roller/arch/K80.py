from .Arch import *

class K80(Arch):
    # compute 3.7
    def __init__(self, para_opt = True):
        super().__init__()
        self.num_level = 2
        # DRAM: memory level 0
        # SMEM: memory level 1
        # REG: memory level 2
        # bandwidth in GBps
        self.bandwidth = [191, 2540] # measured
        # compute throughput in GFLOPS
        self.peak_flops = 4113 # found in hardware spec TODO
        self.limit = []
        self.reg_cap = [32768, 96] # max reg per block(from hw: 65536), max reg per thread TODO 0 get device, 1 what?
        self.smem_cap = [49152] # device query, max shared memory per block TODO get device?
        self.compute_max_core = [26, 26 * 4 * 32]
        self.mem_max_core = [26, 26 * 4 * 32]
        self.para_opt = para_opt

        self.warp_size = 32
        self.compute_sm_partition = [26, 4] # TODO verify?
        self.smem_sm_partition = [26, 4]
        self.compute_block_schedule_way = ["warp", "active block"]
        self.smem_block_schedule_way = ["warp", "active block"]
        self.transaction_size = [32, 256]   # TODO: global memory
        self.glbmem_sm_partition = [26, 32]   # TODO 32: The number of warps per sm when global memory reaches the peak throughput
        self.smem_bank_size = 8
        self.bank_number = 32
        self.compute_capability = 'compute_37'
