from .arch import *


class RTX3090(Arch):
  # compute 8.6  
  def __init__(self, para_opt=True):
    super().__init__()
    self.num_level = 2    
    # DRAM: memory level 0    
    # SMEM: memory level 1    
    # REG: memory level 2    
    # bandwidth in GBps    
    self.bandwidth = [790, 12382] # 12080/80*82    
    # compute throughput in GFLOPS    
    self.peak_flops = 20800 # cutlass perf # 35600 in theory   
    self.peak_tc_flops = 129000 #cutlass perf #142000 in theory   
    self.limit = []
    self.reg_cap = [32768, 96]
    self.smem_cap = [49152]
    self.compute_max_core = [82, 82 * 4 * 32]
    self.mem_max_core = [82, 82 * 4 * 32]
    self.para_opt = para_opt    
    self.warp_size = 32    
    self.compute_sm_partition = [82, 4]
    self.smem_sm_partition = [82, 4]
    self.compute_block_schedule_way = ['warp', 'active block']
    self.smem_block_schedule_way = ['warp', 'active block']
    self.transaction_size = [32, 128]  # in bytes    
    self.glbmem_sm_partition = [82, 32]  # 32: The number of warps per sm when global memory reaches the peak throughput    
    self.smem_bank_size = 4    
    self.bank_number = 32    
    self.compute_capability = 'compute_86'    
    # for active block estimation    
    self.max_active_blocks = 32    
    self.max_smem_usage = 96 * 1024 - 1    
    self.max_threads_per_sm = 1536