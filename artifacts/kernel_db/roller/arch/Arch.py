class Arch:
    def memory_bw(self, mem_level, num_core = -1):
        if num_core == -1:
            return self.bandwidth[mem_level]
        else:
            return self.bandwidth[mem_level] * self.memory_penalty(num_core)

    def _peak_flops(self, num_core = -1):
        if num_core == -1:
            return self.peak_flops
        else:
            return self.peak_flops * self.compute_penalty(num_core)

    def _reg_cap(self, mem_level):
        return self.reg_cap[mem_level]

    def mem_cap(self, mem_level):
        return self.smem_cap[mem_level]

    def _memory_penalty(self, num_core):
        if not self.para_opt: return 1
        min_penalty = 1
        for mx, real in zip(self.mem_max_core, num_core):
            if real is not None:
                import math
                penalty = 1. / math.ceil(real / mx) / (mx / real)
                min_penalty = min(min_penalty, penalty)
        return min_penalty

    def _compute_penalty(self, num_core):
        if not self.para_opt: return 1
        min_penalty = 1
        for mx, real in zip(self.compute_max_core, num_core):
            if real is not None:
                import math
                penalty = 1. / math.ceil(real / mx) / (mx / real)
                min_penalty = min(min_penalty, penalty)
        return min_penalty
    def _smem_bank_size(self):
        return self.smem_bank_size
    
    def _bank_number(self):
        return self.bank_number
