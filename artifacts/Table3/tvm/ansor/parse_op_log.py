import sys
import os
import json
import math

def get(filename):
    best_result = math.inf
    best_step = 0
    with open(filename, "r") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            obj = json.loads(line)
            if "r" in obj:
                result = obj["r"]
            else:
                result = obj["result"]
            # result[0] runtime
            # result[1] error code
            # result[2] compilation time
            if result[1] == 0 and result[0][0] < best_result: # error number equals 0 means no error
                best_result = result[0][0]
                best_step = idx
        return best_result, best_step

def get_log_filename_conv(N, CI, H, W, CO, KH, KW, strides, padding, path):
    return os.path.join(path, "ansor_conv2d_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}.log".format(N, CI, H, W, CO, KH, KW, strides, padding))

def get_log_filename_depthwise(N, CI, H, W, KH, KW, strides, padding, path):
    return os.path.join(path, "ansor_depthwise_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}.log".format(N, CI, H, W, KH, KW, strides, padding))

def get_log_filename_elementwise(t, path, *shape):
    return os.path.join(path, "ansor_{0}_{1}.log".format(t, "_".join([str(s) for s in shape])))

def get_log_filename_matmul(M, K, N, path):
    return os.path.join(path, "ansor_matmul_{0}_{1}_{2}.log".format(M, K, N))

def get_log_filename_pooling(pool_type, path, *shape):
    return os.path.join(path, "ansor_{0}_pooling_{1}.log".format(pool_type, "_".join([str(s) for s in shape])))

def get_log_filename_reduce(path, shape, axis):
    return os.path.join(path, "ansor_{0}_{1}_axis_{2}.log".format("reduction", "_".join([str(s) for s in shape]), axis))

def main():
    log_dir = sys.argv[1]
    op = sys.argv[2]
    path = os.path.join(log_dir, op)

    if op == "conv":
        N, CI, H, W, CO, KH, KW, strides, dilation, padding = [int(s) for s in sys.argv[3:13]]
        log_filename = get_log_filename_conv(N, CI, H, W, CO, KH, KW, strides, padding, path)
    
    if op == "depthwise":
        N, CI, H, W, KH, KW, strides = [int(s) for s in sys.argv[3:10]]
        padding = sys.argv[10]
        log_filename = get_log_filename_depthwise(N, CI, H, W, KH, KW, strides, padding, path)
    
    if op == "elementwise":
       shape = tuple([int(s) for s in sys.argv[4:]])
       t = sys.argv[3]
       log_filename = get_log_filename_elementwise(t, path, *shape)
    
    if op == "matmul":
        batch, in_dim, out_dim = [int(s) for s in sys.argv[3:6]]
        log_filename = get_log_filename_matmul(batch, in_dim, out_dim, path)
    
    if op == "pooling":
        N, CI, H, W, K, strides = tuple([int(s) for s in sys.argv[4:10]])
        padding = sys.argv[10]
        pool_type = sys.argv[3]
        log_filename = get_log_filename_pooling(pool_type, path, N, CI, H, W, K, K, strides, padding)
    
    if op == "reduction":
        shape = tuple([int(s) for s in sys.argv[3:-1]])
        axis = int(sys.argv[-1])
        if axis + 1 < len(shape):
            axis = tuple([x for x in range(axis, len(shape))])
        log_filename = get_log_filename_reduce(path, shape, axis)
    
    print("best runtime: ", get(log_filename)[0] * 1000)

main()