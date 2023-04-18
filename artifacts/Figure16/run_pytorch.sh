#!/bin/bash

LOG_DIR=${ARTIFACT_ROOT}/reproduce_results/Figure16/pytorch
mkdir -p ${LOG_DIR}
cd ${ARTIFACT_ROOT}/../baseline

cd resnet18
python3 resnet18_pytorch.py --bs 1 --platform V100 2>&1 | tee ${LOG_DIR}/blockdrop.noskip.log
nvprof --profile-from-start off python3 resnet18_pytorch.py --bs 1 --platform V100 2>&1 | tee ${LOG_DIR}/blockdrop.noskip.nvprof.log
cd ..

cd blockdrop
python3 blockdrop_pytorch.py --bs 1 --platform V100 --overhead_test --unroll 2>&1 | tee ${LOG_DIR}/blockdrop.unroll.log
nvprof --profile-from-start off python3 blockdrop_pytorch.py --bs 1 --platform V100 --overhead_test --unroll 2>&1 | tee ${LOG_DIR}/blockdrop.unroll.nvprof.log
python3 blockdrop_pytorch.py --bs 1 --platform V100 --overhead_test --fix 2>&1 | tee ${LOG_DIR}/blockdrop.fix.log
nvprof --profile-from-start off python3 blockdrop_pytorch.py --bs 1 --platform V100 --overhead_test --fix 2>&1 | tee ${LOG_DIR}/blockdrop.fix.nvprof.log
cd ..

cd resnet101
python3 resnet101_pytorch.py --bs 1 --platform V100 2>&1 | tee ${LOG_DIR}/skipnet.noskip.log
nvprof --profile-from-start off python3 resnet101_pytorch.py --bs 1 --platform V100 2>&1 | tee ${LOG_DIR}/skipnet.noskip.nvprof.log
cd ..

cd skipnet
python3 skipnet_pytorch.py --bs 1 --platform V100 --overhead_test --unroll 2>&1 | tee ${LOG_DIR}/skipnet.unroll.log
nvprof --profile-from-start off python3 skipnet_pytorch.py --bs 1 --platform V100 --overhead_test --unroll 2>&1 | tee ${LOG_DIR}/skipnet.unroll.nvprof.log
python3 skipnet_pytorch.py --bs 1 --platform V100 --overhead_test --fix 2>&1 | tee ${LOG_DIR}/skipnet.fix.log
nvprof --profile-from-start off python3 skipnet_pytorch.py --bs 1 --platform V100 --overhead_test --fix 2>&1 | tee ${LOG_DIR}/skipnet.fix.nvprof.log
cd ..
