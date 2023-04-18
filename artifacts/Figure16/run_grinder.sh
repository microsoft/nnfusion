#!/bin/bash

LOG_DIR=${ARTIFACT_ROOT}/reproduce_results/Figure16/grinder
mkdir -p ${LOG_DIR}
cd ${ARTIFACT_ROOT}/models

python3 resnet18.py --platform V100 --bs 1 --no-torch --measure 2>&1 | tee ${LOG_DIR}/blockdrop.noskip.log
nvprof --profile-from-start off python3 resnet18.py --platform V100 --bs 1 --no-torch --measure 2>&1 | tee ${LOG_DIR}/blockdrop.noskip.nvprof.log
python3 blockdrop.py --bs 1 --no-torch --overhead_test --unroll --measure 2>&1 | tee $LOG_DIR/blockdrop.unroll.log
nvprof --profile-from-start off python3 blockdrop.py --bs 1 --no-torch --overhead_test --unroll --measure 2>&1 | tee $LOG_DIR/blockdrop.unroll.nvprof.log
python3 blockdrop.py --bs 1 --no-torch --overhead_test --fix --measure 2>&1 | tee $LOG_DIR/blockdrop.fix.log
nvprof --profile-from-start off python3 blockdrop.py --bs 1 --no-torch --overhead_test --fix --measure 2>&1 | tee $LOG_DIR/blockdrop.fix.nvprof.log

python3 resnet101.py --platform V100 --bs 1 --no-torch --measure 2>&1 | tee ${LOG_DIR}/skipnet.noskip.log
nvprof --profile-from-start off python3 resnet101.py --platform V100 --bs 1 --no-torch --measure 2>&1 | tee ${LOG_DIR}/skipnet.noskip.nvprof.log
python3 skipnet.py --bs 1 --no-torch --overhead_test --unroll --measure 2>&1 | tee $LOG_DIR/skipnet.unroll.log
nvprof --profile-from-start off python3 skipnet.py --bs 1 --no-torch --overhead_test --unroll --measure 2>&1 | tee $LOG_DIR/skipnet.unroll.nvprof.log
python3 skipnet.py --bs 1 --no-torch --overhead_test --fix --measure 2>&1 | tee $LOG_DIR/skipnet.fix.log
nvprof --profile-from-start off python3 skipnet.py --bs 1 --no-torch --overhead_test --fix --measure 2>&1 | tee $LOG_DIR/skipnet.fix.nvprof.log