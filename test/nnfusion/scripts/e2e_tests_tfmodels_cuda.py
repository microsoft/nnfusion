# Microsoft (c) 2019, MSRA/NNFUSION Team
# Author: wenxh
# This script is to be used as batch system intergration test in Azure Build Agent
import os
import sys
import subprocess
import logging
import numpy as np

ground_truth = {"frozen_random-weights_bert_large.pb":
                "0.001335 0.001490 0.000675 0.002558 0.000761 0.001435 0.000518 0.001516 0.000738 0.001183  .. (size = 1001, ends with 0.000281);",
                "frozen_alexnet_infer_batch_1.pb": "0.000914 -0.030341 -0.006662 -0.010238 0.014080 0.024311 0.006832 -0.035370 0.017920 0.038856  .. (size = 1001, ends with 0.022597);",
                "frozen_resnet50_infer_batch_1.pb": "-0.001597 0.030608 -0.002212 0.037812 0.030037 0.039713 -0.006352 0.051142 0.016946 -0.009263  .. (size = 1001, ends with 0.043752);",
                "frozen_vgg11_infer_batch_1.pb": "-0.003832 -0.008819 0.004029 -0.003441 0.012382 -0.003776 0.001756 -0.014141 -0.005059 -0.001504  .. (size = 1001, ends with 0.008471);",
                "frozen_inception3_infer_batch_1.pb": "-0.000079 -0.000875 -0.000871 -0.000491 0.000316 -0.000246 0.000187 0.000502 -0.000710 -0.000311  .. (size = 1001, ends with -0.000542);"}

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

models = sys.argv[1]
nnfusion_cli = sys.argv[2]

# check
if not(os.path.exists(models) and os.path.exists(nnfusion_cli)):
    logging.error("NNFusion cli or model folder is not existed.")
    exit(1)

for pbfile in os.listdir(models):
    if not pbfile.endswith("pb"):
        continue
    if pbfile not in ground_truth.keys():
        continue
    os.system("rm -rf nnfusion_rt")
    logging.info("Compiling " + pbfile)
    os.system("%s %s -f tensorflow -b nnfusion >> nnfusion.log" %
              (nnfusion_cli, os.path.join(models, pbfile)))
    if not os.path.exists("nnfusion_rt/cuda_codegen/nnfusion_rt.cu"):
        logging.error("Failed at nnfusion compiling phase.")
        exit(2)
    os.system("cd nnfusion_rt/cuda_codegen/ && cmake . >> cmake.log && make -j 2>&1 >> cmake.log")
    if not os.path.exists("nnfusion_rt/cuda_codegen/main_test"):
        logging.error("Failed at nvcc compiling phase.")
        exit(3)
    os.system("cd nnfusion_rt/cuda_codegen/ && ./main_test > result.txt")
    if not os.path.exists("nnfusion_rt/cuda_codegen/result.txt"):
        logging.error("Failed at nvcc compiling phase.")
        exit(4)
    result_file = open("nnfusion_rt/cuda_codegen/result.txt")
    results = result_file.readlines()
    if len(results) >= 2 : #or results[1].strip() != ground_truth[pbfile]:
        a_data = [float(v.strip()) for v in results[1].split("..")[0].strip().split(" ")]
        b_data = [float(v.strip()) for v in ground_truth[pbfile].split("..")[0].strip().split(" ")]
        if not np.allclose(a_data, b_data, rtol=1.e-4, atol=1.e-4):
            appended = "."
            if len(results) > 1:
                appended = "Expected: %s vs Output: %s." % (
                    ground_truth[pbfile], results[1])
            logging.error("%s has wrong result - "%pbfile + appended)
            exit(5)
        else:
            print("%s has right result!"%pbfile)
    else:
        exit(6)
    os.system("rm -rf nnfusion_rt")
print("All Done!.")
exit(0)
