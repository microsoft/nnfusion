#!/bin/bash

LOG_DIR=${ARTIFACT_ROOT}/reproduce_results/Figure18/pytorch
mkdir -p ${LOG_DIR}
cd ${ARTIFACT_ROOT}/baseline
cd rae
python3 rae_pytorch.py --bs 1 --platform V100 --overhead_test --unroll 2>&1 | tee ${LOG_DIR}/rae.unroll.log
nvprof --profile-from-start off python3 rae_pytorch.py --bs 1 --platform V100 --overhead_test --unroll 2>&1 | tee ${LOG_DIR}/rae.unroll.nvprof.log
python3 rae_pytorch.py --bs 1 --platform V100 --overhead_test --fix 2>&1 | tee ${LOG_DIR}/rae.fix.log
nvprof --profile-from-start off python3 rae_pytorch.py --bs 1 --platform V100 --overhead_test --fix 2>&1 | tee ${LOG_DIR}/rae.fix.nvprof.log
cd ..
cd ..