import tvm

# class V100:
#     def __init__(self):
#         self.reg_cap = 32768
#         self.smem_cap = 49152
#         self.compute_max_core = 80
#         self.warp_size = 32
#         self.sm_partition = 4
#         self.transaction_size = [32, 128]   # in bytes
#         self.max_smem_usage = 96 * 1024
#         self.bandwidth = [750, 12080]
#         self.platform = "CUDA"
#         self.compute_capability = "70"
#         self.target = tvm.target.cuda(model="V100", arch="sm_70")

class V100:
    def __init__(self):
        self.reg_cap = 32768
        self.smem_cap = 49152
        self.compute_max_core = 108
        self.warp_size = 32
        self.sm_partition = 4
        self.transaction_size = [32, 128]   # in bytes
        self.max_smem_usage = 164 * 1024
        self.bandwidth = [1319, 16308]
        self.platform = "CUDA"
        self.cutlass_mma = [16, 8, 16]
        self.compute_capability = "80"
        self.target = tvm.target.cuda(model="A100", arch="sm_80")
