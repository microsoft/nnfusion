from roller_kernels import run

if __name__ == '__main__':
    run("BatchMatMul[64,256;8,256,256;8,64,256floatfloatfloat]", "V100")