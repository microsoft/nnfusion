#!/bin/bash

LOG_DIR=${ARTIFACT_ROOT}/reproduce_results/Figure18/grinder
mkdir -p ${LOG_DIR}
cd ${ARTIFACT_ROOT}/models

python3 rae-unroll.py --platform V100 --bs 1 --no-torch --measure 2>&1 | tee $LOG_DIR/rae.unroll.log
nvprof --profile-from-start off python3 rae-unroll.py --measure 2>&1 | tee $LOG_DIR/rae.unroll.nvprof.log
python3 rae.py --platform V100 --bs 1 --no-torch --overhead_test --measure 2>&1 | tee $LOG_DIR/rae.fix.log
nvprof --profile-from-start off python3 rae.py --platform V100 --bs 1 --no-torch --measure 2>&1 | tee $LOG_DIR/rae.fix.nvprof.log
