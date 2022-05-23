

class TileInfo():
    def __init__(self, footprint, smem_cost, block_per_SM, num_wave) -> None:
        self.footprint = footprint
        self.smem_cost = smem_cost
        self.block_per_SM = block_per_SM
        self.num_wave = num_wave

    def valid(self):
        return self.num_wave >= 0
