# Microsoft (c) 2019, MSRA/NNFUSION Team
# Author: wenxh
# This script is to be used as batch system intergration test in Azure Build Agent
import os
import sys
import subprocess
import logging
import numpy as np

# todo(wenxh): replace those accurate result.
ground_truth_str = '''Result_1077_0:
1.544510e-03 5.187932e-04 8.498678e-04 1.672361e-03 5.371213e-04 4.475404e-04 1.268131e-03 5.944323e-04 1.431170e-03 3.446568e-04  .. (size = 1024, ends with 5.337071e-04);
Result_1078_0:
6.931930e+00  .. (size = 1, ends with 6.931930e+00);
Result_1125_0:
-1.907903e-04 1.271152e-04 1.543891e-05 -1.271294e-04 1.068962e-05 -1.242892e-04 1.388441e-04 -8.514979e-05 -1.445845e-04 3.206256e-05  .. (size = 524288, ends with 0.000000e+00);
'''.strip().split("\n")

def extract_data(strs):
    data = list()
    for i in range(1, len(strs), 2):
        data.append([float(v.strip())
                     for v in strs[i].strip().split("..")[0].strip().split(" ")])
    return data

def all_allclose(a, b):
    cnt = 0
    for u in a:
        flag = False
        for v in b:
            if np.allclose(u, v, rtol=1.e-4, atol=1.e-4):
                flag = True
        if not flag:
            print("Mismatch#%d: %s"%(cnt, u))
            return False
        cnt += 1
    return True

# Inputs:
# - mfolder: model folder
# - nnfusion: nnfusion place
# generatecode : $nnfusion $mfolder/*.pb -f tensorflow -b nnfusion_engine
# cmake : cd nnfusion_rt/cuda && cmake . && make -j
# test : cd nnfusion_rt/cuda && main_test -> compare result
# clean : rm -rf nnfusion_rt


# error args, not executing command
if not os.path.exists("/usr/local/cuda/bin/nvcc"):
    logging.info("NVCC is not existed, thus skip the test.")
    exit(0)

if len(sys.argv) != 3:
    logging.error("Script doesn't have right arguments.")
    exit(1)
if not sys.argv[2].endswith("nnfusion"):
    logging.error("NNFusion cli should named \"nnfusion\"")
    exit(1)

pbfile = sys.argv[1]
nnfusion_cli = sys.argv[2]

# check
if not(os.path.exists(pbfile) and os.path.exists(nnfusion_cli)):
    logging.error("NNFusion cli or model folder is not existed.")
    exit(1)

os.system("rm -rf nnfusion_rt")
logging.info("Compiling " + pbfile)
os.system("%s %s -f tensorflow -b nnfusion >> nnfusion.log" %
            (nnfusion_cli, pbfile))
if not os.path.exists("nnfusion_rt/cuda_codegen/nnfusion_rt.cu"):
    logging.error("Failed at nnfusion compiling phase.")
    exit(2)
os.system(
    "cd nnfusion_rt/cuda_codegen/ && cmake . >> cmake.log && make -j 2>&1 >> cmake.log")
if not os.path.exists("nnfusion_rt/cuda_codegen/main_test"):
    logging.error("Failed at nvcc compiling phase.")
    exit(3)
os.system("cd nnfusion_rt/cuda_codegen/ && ./main_test > result.txt")
if not os.path.exists("nnfusion_rt/cuda_codegen/result.txt"):
    logging.error("Failed at nvcc compiling phase.")
    exit(4)
result_file = open("nnfusion_rt/cuda_codegen/result.txt")
results = result_file.readlines()
if len(results) >= len(ground_truth_str):  # or results[1].strip() != ground_truth[pbfile]:
    a_data = extract_data(ground_truth_str[:6])
    b_data = extract_data(results[:6])
    if not all_allclose(b_data, a_data):
        logging.error("%s has wrong result" % pbfile)
        exit(5)
    else:
        print("%s has right result!" % pbfile)
else:
    exit(6)
os.system("rm -rf nnfusion_rt")
print("All Done!.")
exit(0)
