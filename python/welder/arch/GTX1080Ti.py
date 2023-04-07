import tvm

class GTX1080Ti:
    def __init__(self):
        self.reg_cap = 32768
        self.smem_cap = 49152
        self.compute_max_core = 28
        self.warp_size = 32
        self.sm_partition = 4
        self.transaction_size = [32, 128]   # in bytes
        self.max_smem_usage = 96 * 1024
        self.bandwidth = [750, 12080]
        self.platform = "CUDA"
        self.compute_capability = "61"
        self.target = tvm.target.cuda(model="1080ti", arch="sm_61")
