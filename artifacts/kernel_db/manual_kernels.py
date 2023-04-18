from db import save_to_db
import os
import re

kernels = {
    "AvgPool[64,1,8,8;64,1,1,1floatfloatShape{8, 8}Strides{8, 8}Shape{0, 0}Shape{0, 0}]": "avgpool_64_1_1_8_8_S8P0", # blockdrop bs1
    "AvgPool[1,64,8,8;1,64,1,1floatfloatShape{8, 8}Strides{8, 8}Shape{0, 0}Shape{0, 0}]": "avgpool_64_1_1_8_8_S8P0", # blockdrop bs1 pt + our kernel
    "AvgPool[64,64,8,8;64,64,1,1floatfloatShape{8, 8}Strides{8, 8}Shape{0, 0}Shape{0, 0}]": "avgpool_4096_1_1_8_8_S8P0", # blockdrop bs64
    "Dot[1,64;10,64;1,10floatfloatfloat01]": "dot_1_64_10_64_1_10_01", # blockdrop bs1
    "BatchMatMul[1,256;4,256,256;4,1,256floatfloatfloat]": "bmm_4_1_256", # lstm bs1
    "BatchMatMul[1,256;8,256,256;8,1,256floatfloatfloat]": "bmm_8_1_256", # nasrnn bs1
    "AvgPool[256,1,56,56;256,1,1,1floatfloatShape{56, 56}Strides{56, 56}Shape{0, 0}Shape{0, 0}]": "avgpool_256_1_1_56_56_S56P0", # skipnet bs1
    "AvgPool[1,256,56,56;1,256,1,1floatfloatShape{56, 56}Strides{56, 56}Shape{0, 0}Shape{0, 0}]": "avgpool_256_1_1_56_56_S56P0", # skipnet bs1 pt + our kernel
    "Convolution[256,1,1,1;10,256,1,1;10,1,1,1floatfloatfloatLayout{CNHW}Strides{1, 1}Strides{1, 1}CoordinateDiff{0, 0}]": "dot_nt_1_256_10_256_1_10", # skipnet bs1
    "Convolution[1,256,1,1;10,256,1,1;10,1,1,1floatfloatfloatLayout{NCHW2CNHW}Strides{1, 1}Strides{1, 1}CoordinateDiff{0, 0}]": "dot_nt_1_256_10_256_1_10", # skipnet bs1 pt + our kernel
    "BatchMatMul[1,10;4,10,10;4,1,10floatfloatfloat]": "bmm_4_1_10", # skipnet bs1
    "Convolution[1,10,1,1;1,10,1,1;1,1,1,1floatfloatfloatLayout{NCHW2CNHW}Strides{1, 1}Strides{1, 1}CoordinateDiff{0, 0}]": "dot_1_10_1_10_1_1_01", # skipnet bs1
    "Convolution[1,10,1,1;1,10,1,1;1,1,1,1floatfloatfloatLayout{NCHW}Strides{1, 1}Strides{1, 1}CoordinateDiff{0, 0}]": "dot_1_10_1_10_1_1_01", # skipnet bs1
    "AvgPool[512,1,28,28;512,1,1,1floatfloatShape{28, 28}Strides{28, 28}Shape{0, 0}Shape{0, 0}]": "avgpool_512_1_1_28_28_S28P0", # skipnet bs1
    "AvgPool[1,512,28,28;1,512,1,1floatfloatShape{28, 28}Strides{28, 28}Shape{0, 0}Shape{0, 0}]": "avgpool_512_1_1_28_28_S28P0", # skipnet bs1 pt + our kernel
    "Convolution[512,1,1,1;10,512,1,1;10,1,1,1floatfloatfloatLayout{CNHW}Strides{1, 1}Strides{1, 1}CoordinateDiff{0, 0}]": "dot_1_512_10_512_1_10_01", # skipnet bs1
    "Convolution[1,512,1,1;10,512,1,1;10,1,1,1floatfloatfloatLayout{NCHW2CNHW}Strides{1, 1}Strides{1, 1}CoordinateDiff{0, 0}]": "dot_1_512_10_512_1_10_01", # skipnet bs1 pt + our kernel
    "AvgPool[1024,1,14,14;1024,1,1,1floatfloatShape{14, 14}Strides{14, 14}Shape{0, 0}Shape{0, 0}]": "avgpool_1024_1_1_14_14_S14P0", # skipnet bs1
    "AvgPool[1,1024,14,14;1,1024,1,1floatfloatShape{14, 14}Strides{14, 14}Shape{0, 0}Shape{0, 0}]": "avgpool_1024_1_1_14_14_S14P0", # skipnet bs1 pt + our kernel
    "Convolution[1024,1,1,1;10,1024,1,1;10,1,1,1floatfloatfloatLayout{CNHW}Strides{1, 1}Strides{1, 1}CoordinateDiff{0, 0}]": "dot_1_1024_10_1024_1_10_01", # skipnet bs1
    "Convolution[1,1024,1,1;10,1024,1,1;10,1,1,1floatfloatfloatLayout{NCHW2CNHW}Strides{1, 1}Strides{1, 1}CoordinateDiff{0, 0}]": "dot_1_1024_10_1024_1_10_01", # skipnet bs1 pt + our kernel
    "AvgPool[2048,1,7,7;2048,1,1,1floatfloatShape{7, 7}Strides{7, 7}Shape{0, 0}Shape{0, 0}]": "avgpool_2048_1_1_7_7_S7P0", # skipnet bs1
    "AvgPool[1,2048,7,7;1,2048,1,1floatfloatShape{7, 7}Strides{7, 7}Shape{0, 0}Shape{0, 0}]": "avgpool_2048_1_1_7_7_S7P0", # skipnet bs1 pt + our kernel
    "Convolution[2048,1,1,1;10,2048,1,1;10,1,1,1floatfloatfloatLayout{CNHW}Strides{1, 1}Strides{1, 1}CoordinateDiff{0, 0}]": "dot_1_2048_10_2048_1_10_01", # skipnet bs1
    "Convolution[1,2048,1,1;10,2048,1,1;10,1,1,1floatfloatfloatLayout{NCHW2CNHW}Strides{1, 1}Strides{1, 1}CoordinateDiff{0, 0}]": "dot_1_2048_10_2048_1_10_01", # skipnet bs1
    "Dot[1,1024;512,1024;1,512floatfloatfloat01]": "dot_1_1024_512_1024_1_512", # rae bs1
    "Dot[1,256;3797,256;1,3797floatfloatfloat01]": "dot_1_256_3797_256_1_3797_01", # seq2seq bs1
    "BatchMatMul[1,12,1,64;12,64,64;1,12,1,64floatfloatfloat]": "bmm_12_1_64_64", # attention bs1
    "BatchMatMul[1,12,1,64;1,12,64,64;1,12,1,64floatfloatfloat]": "bmm_12_1_64_64", # attention bs1
}

def save_one(identifier):
    file_name = kernels[identifier] + ".cu"
    with open(os.path.join("manual_kernels", file_name)) as f:
        while True:
            st = f.readline()
            if r"// %%%" in st:
                break
        kernel_code = []
        while True:
            st = f.readline()
            if r"// %%%" in st:
                break
            kernel_code.append(st)
        kernel_code = "".join(kernel_code)
        
        while True:
            st = f.readline()
            if r"// +++" in st:
                break
        grid = f.readline()
        grid = re.search("(\d+), (\d+), (\d+)", grid).groups()
        grid = tuple(int(x) for x in grid)
        block = f.readline()
        block = re.search("(\d+), (\d+), (\d+)", block).groups()
        block = tuple(int(x) for x in block)
        save_to_db(identifier, kernel_code, grid, block)
        

def save_all():
    for id in kernels:
        print(id)
        save_one(id)

if __name__ == "__main__":
    save_all()
