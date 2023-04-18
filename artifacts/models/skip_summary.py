import numpy as np

def read_bin(s):
    with open(s + ".shape") as f: shape = tuple((int(x) for x in f.read().strip().split(" ")))
    tensor = np.fromfile(s + ".bin", dtype=np.float32).reshape(shape)
    return tensor


def analyze_skip(tensor, bs, name):
    n_run = 100
    x = tensor[:bs * n_run].reshape(n_run, bs, -1) # n_run, bs, skip_or_not
    x = x.sum(axis=1)
    skip = (x == 0)
    print(name, "batch_size", bs, "need_run", skip.sum(), "total", skip.size, "ratio", skip.sum() / skip.size)
    


skipnet_actions = read_bin("../data/skipnet/actions")
analyze_skip(skipnet_actions, 1, "skipnet")
analyze_skip(skipnet_actions, 64, "skipnet")

blockdrop_probs = read_bin("../data/blockdrop/probs")
blockdrop_cond = blockdrop_probs > 0.5
analyze_skip(blockdrop_cond, 1, "blockdrop")
analyze_skip(blockdrop_cond, 64, "blockdrop")
