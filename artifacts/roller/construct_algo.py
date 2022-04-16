import numpy as np

roller = Roller()
device = Device()
model = CostModel()

TOPK = 10
top_results = []


def IncreaseTile(tile, steps, mem, res):
    reuse_score = roller.DataReuseScore(mem, tile, steps)
    dim_idx = np.argsort(reuse_score)
    for dim in reversed(dim_idx):
        new_tile = tile
        new_tile[dim] = steps[mem][dim].next()
        if model.Footprint(new_tile, mem) > mem.Capacity:
            continue
        elif not model.IsPeakComputeTile(new_tile, mem) or model.MemLatency(new_tile, mem) > model.ComputeLatency(new_tile):
            IncreaseTile(new_tile, steps, mem, res)
        else:
            res.append(new_tile)
            if mem.IsLastLayer():
                top_results.append(res)
                if len(top_results) > TOPK:
                    exit()
            else:
                next_mem = device.NextMemLayer(mem)
                new_steps = roller.GetAlignedSteps(next_mem)
                IncreaseTile(new_tile, new_steps, next_mem, res)

def construct_algo(expr):
    for mem in range(device.num_level):
        steps[mem] = roller.GetAlignedSteps(mem)
    init_tile = np.ones(expr.axis_num())
    res_tiles = []
    IncreaseTile(init_tile, steps, mem, res_tiles)
