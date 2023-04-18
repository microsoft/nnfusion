class Schedule():
    def __init__(self, dim_size, spatial_axis = [], reduce_axis = [], 
                active_blocks_per_sm = 1, compute_peak_performance = 14000, glbmem_bandwidth = 750, smem_bandwidth = 12080):
        # schedule structure:
        # mem_level -> tile_size
        # mem_level -> {(reduction axis name) -> (reduction axis len)}
        self._size = {}
        self._reduction_size = {}
        self.dim_size = dim_size
        self.spatial_axis = spatial_axis
        self.reduce_axis = reduce_axis
        self.fused_spatial_axis_base_delta = []
        self.fused_reduce_axis_base_delta = []
        """
        multi-level warp performance for cost model
        e.g. on GPUs:
            0 -> DRAM throughput (GBps)
            1 -> shared memory throughput (GBps)
            2 -> compute throughput (GFLOPS)
        """
        self.active_blocks_per_sm = active_blocks_per_sm
        self.compute_peak_performance = compute_peak_performance
        self.glbmem_bandwidth = glbmem_bandwidth
        self.smem_bandwidth = smem_bandwidth

    def update_tile(self, mem_level, dim, reduction_dict):
        # add both the tile size & reduction size
        if dim != None:
            if isinstance(dim, list):
                dim = tuple(dim)
            assert len(list(dim)) == self.dim_size
            self._size[mem_level] = dim
        if reduction_dict != None:
            self._reduction_size[mem_level] = {}
            for key in reduction_dict:
                assert key in self.reduce_axis
                self._reduction_size[mem_level][key] = reduction_dict[key]

    def add_tile(self, mem_level, dim, reduction_dict={}):
        # add both the tile size & reduction size
        assert not mem_level in self._size
        self.update_tile(mem_level, dim, reduction_dict)

    def delete_tile(self, mem_level):
        del self._size[mem_level]
        del self._reduction_size[mem_level]

    def fuse_axis(self, axis_group, base, delta):
        # axis_group: whether fusing reduce regular axes
        # base: the index of first axis to be fused
        # delta: the number of axes to be fused
        new_axis_name = ""
        if axis_group == "reduce":
            self.fused_reduce_axis_base_delta.append((base, delta))
            new_axis_name = self.reduce_axis[base]
            self.reduce_axis.pop(base)
            for x in range(delta - 1):
                new_axis_name = new_axis_name + "." + self.reduce_axis[base] + ".fused"
                self.reduce_axis.pop(base)
            self.reduce_axis.insert(base, new_axis_name)
        else:
            self.fused_spatial_axis_base_delta.append((base, delta))
            new_axis_name = ""
            new_axis_name = self.spatial_axis[base]
            self.spatial_axis.pop(base)
            for x in range(delta - 1):
                new_axis_name = new_axis_name + "." + self.spatial_axis[base] + ".fused"
                self.spatial_axis.pop(base)
            self.spatial_axis.insert(base, new_axis_name)
            self.dim_size -= delta - 1
        return new_axis_name

    def get_tile(self, mem_level):
        # return both the tile size & reduction size
        if mem_level not in self._size:
            return None
        return self._size[mem_level], self._reduction_size[mem_level]

    def subtile_count(self, low, high):
        # lower level tile is larger
        assert low in self._size
        assert high in self._size
        dim_size = len(self._size[low])
        ret = 1
        for i in range(dim_size):
            ret = ret * ((self._size[low][i] - 1) // self._size[high][i] + 1)
        return ret

    def copy(self):
        new_copy = Schedule(self.dim_size, self.spatial_axis.copy(), self.reduce_axis.copy())
        new_copy._size = self._size.copy()
        new_copy._reduction_size = self._reduction_size.copy()
        new_copy.fused_spatial_axis_base_delta = self.fused_spatial_axis_base_delta.copy()
        new_copy.fused_reduce_axis_base_delta = self.fused_reduce_axis_base_delta.copy()
        return new_copy

    def to_codegen_dict(self):
        ret = {}
        for raxis_name in self.reduce_axis:
            ret[raxis_name] = []
            for i in range(len(self._size) - 1):
                ret[raxis_name].append((self._reduction_size[i][raxis_name] - 1) // self._reduction_size[i + 1][raxis_name] + 1)
        self.thread_reg_size = [1 for _ in range(self.dim_size)]
        axis_id = 0
        for axis_name in self.spatial_axis:
            ret[axis_name] = []
            for i in range(len(self._size) - 1):
                ret[axis_name].append((self._size[i][axis_id] - 1) // self._size[i + 1][axis_id] + 1)
            axis_id += 1
        return ret

    def dump_to_string(self):
        ret = ""
        for i in self._size:
            ret = ret + "level{}: [tile size: {}]".format(i, self._size[i])
            ret = ret + "[reduction_size: {}]; ".format(self._reduction_size[i])
        return ret
