from .arch import *

class XboxSeriesXDX(Arch):
  # RDNA2  
  def __init__(self, para_opt=True):
    super().__init__()
    self.num_level = 2    # DRAM: memory level 0    # SMEM: memory level 1    # REG: memory level 2    
    # bandwidth in GBps 
    self.bandwidth = [500, 8000] # 12080/80*82    
    # compute throughput in GFLOPS    
    self.peak_flops = 9300 # cutlass perf # 35600 in theory    
    self.limit = []
    self.reg_cap = [20000, 70]
    self.smem_cap = [32768] # DirectX limit    
    self.compute_max_core = [52]
    self.warp_size = 32    
    self.compute_sm_partition = [52, 4]
    self.smem_sm_partition = [52, 4]
    self.compute_block_schedule_way = ['warp']
    self.smem_block_schedule_way = ['warp']
    self.transaction_size = [32, 128]  # in bytes    
    self.glbmem_sm_partition = [52, 1]  # 32: The number of warps per sm when global memory reaches the peak throughput   
    self.smem_bank_size = 4