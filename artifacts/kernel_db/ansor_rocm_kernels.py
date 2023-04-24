# use the final.cu tuned on V100

import os
from db import save_to_db
from test_config import *
from tvm import te, auto_scheduler

def save(identifier, kernel_dir):
    with open(os.path.join("ansor_kernels", kernel_dir, "final.cu")) as f:
        final = f.readlines()
        # cut until a line with '}'
        best_source = "".join(final[:final.index("}\n") + 1])+"\n"
        best_grid_size = None
        best_block_size = None
        for line in final:
            if line.startswith("dim3 grid"):
                assert best_grid_size is None
                best_grid_size = line[len("dim3 grid"):].strip()
                best_grid_size = best_grid_size.replace("(", "").replace(");", "").split(", ")
                best_grid_size = tuple([int(x) for x in best_grid_size])
            if line.startswith("dim3 block"):
                assert best_block_size is None
                best_block_size = line[len("dim3 block"):].strip()
                best_block_size = best_block_size.replace("(", "").replace(");", "").split(", ")
                best_block_size = tuple([int(x) for x in best_block_size])
        assert best_grid_size is not None
        assert best_block_size is not None
    save_to_db(identifier, best_source, best_grid_size, best_block_size, device_type="ROCM_GPU")

save("BatchMatMul[1,12,64,64;1,12,64,1;1,12,64,1floatfloatfloat]", "batch_matmul_4d_4d_expr_1_12_64_1_64_ansor")
