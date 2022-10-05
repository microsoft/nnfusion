from .Arch import *

class IPU(Arch):
    def __init__(self, para_opt = True):
        super().__init__()
        self.num_level = 2
        # Remote Nodes: memory level 0
        # Local Node: memory level 1
        # REG: memory level 2
        # bandwidth in TBps when channels are fully utilized
        self.bandwidth = [7.68, 7.59]
        self.peak_flops = 18.9 # in TFLOPS, measured if all cores are fully utilized
        self.reg_cap = [96, 16] # unknown for now
        self.smem_cap = [262144] # this is the limit of both smem + reg tile
        self.Transaction_size = [4, 4] # in bytes
        self.num_node = 1216
        self.threads_per_node = 6
        self.para_opt = para_opt
