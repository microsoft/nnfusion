
def size_of(dtype):
    if dtype == "float":
        return 4
    if dtype == "half":
        return 2

class OpBase():
    def subtensor_dim(self, tile_dim, reduction_size, mem_level, tile_tensor = "output"):
        # If the given tile is output, returns:
        #     For each tensor, the size of subtensors required to compute the tile
        # If the given tile is input, returns:
        #     the size of output subtensor that takes this tile as input
        #     and the input tile required to compute the output subtensor
        # returns a dict:
        #     tensor name: str -> (dimension, axis_tags).
        #     A axis tag denotes the axis name and whether it is reduce
        #
        raise NotImplementedError

    def memory_footprint(self, tile_dim, reduction_size, mem_level, tile_tensor = "output", padding = 0):
        #
        # return the memory footprint in byte of a tile given its size and the reduction steps
        # dim: list[int]
        # reduction_size: {str -> int} (axis_name to axis_dim)
        #
        subtensors = self.subtensor_dim(tile_dim, reduction_size, mem_level, tile_tensor)
        ret = 0
        for st_name in subtensors:
            if st_name == tile_tensor:
                continue
            subtensor = subtensors[st_name]
            area = 1
            for i in range(len(subtensor)):
                area *= subtensor[i]
            ret += area
        return ret * size_of(self.input_type) # for now assume all elements are sp floats

    def compute_workload(self, tile_dim, reduction_size, tile_tensor = "output"):
        # Given a tile size, returns the number of FLOs related to this tile
        raise NotImplementedError

    def memory_workload(self, tile_dim, reduction_size, mem_level, tile_tensor = "output"):
        # Given a tile size, returns the total bytes of subtensors related to this tile
        raise NotImplementedError

    def reg_usage(self, tile_dim, tile_tensor = "output"):
        # Given a tile size, returns the number of registers that this tile needs to use
        raise NotImplementedError

    def constant_ratio(self):
        # returns whether the ratio between memory and compute is constant regardless of the tile size
        raise NotImplementedError
    
    def get_grid_size(self, tile_dim, tile_tensor = "output"):
        # Given a tile size, return the grid size of this op
        raise NotImplementedError
    
    def uni_schedule(self, saxis_names, raxis_names):
        tile = [1 for _ in range(len(saxis_names))]
        rstep = {raxis_name:1 for raxis_name in raxis_names}
        return tile, rstep
        