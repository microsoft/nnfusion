#!/bin/bash

LOG_DIR=${ARTIFACT_ROOT}/reproduce_results/Figure15/sys
mkdir -p ${LOG_DIR}
cd ${ARTIFACT_ROOT}/models

python3 lstm-unroll.py --measure 2>&1 | tee $LOG_DIR/lstm.unroll.log
nvprof --profile-from-start off python3 lstm-unroll.py --measure 2>&1 | tee $LOG_DIR/lstm.unroll.nvprof.log
python3 lstm.py --platform V100 --bs 1 --no-torch --measure 2>&1 | tee $LOG_DIR/lstm.fix.log
nvprof --profile-from-start off python3 lstm.py --platform V100 --bs 1 --no-torch --measure 2>&1 | tee $LOG_DIR/lstm.fix.nvprof.log

python3 nasrnn-unroll.py --measure 2>&1 | tee $LOG_DIR/nasrnn.unroll.log
nvprof --profile-from-start off python3 nasrnn-unroll.py --measure 2>&1 | tee $LOG_DIR/nasrnn.unroll.nvprof.log
python3 nasrnn.py --platform V100 --bs 1 --no-torch --measure 2>&1 | tee $LOG_DIR/nasrnn.fix.log
nvprof --profile-from-start off python3 nasrnn.py --platform V100 --bs 1 --no-torch --measure 2>&1 | tee $LOG_DIR/nasrnn.fix.nvprof.log


python3 attention-unroll.py --measure 2>&1 | tee $LOG_DIR/attention.unroll.log
nvprof --profile-from-start off python3 attention-unroll.py --measure 2>&1 | tee $LOG_DIR/attention.unroll.nvprof.log
python3 attention.py --platform V100 --bs 1 --no-torch --measure 2>&1 | tee $LOG_DIR/attention.fix.log
nvprof --profile-from-start off python3 attention.py --platform V100 --bs 1 --no-torch --measure 2>&1 | tee $LOG_DIR/attention.fix.nvprof.log

python3 seq2seq.py --platform V100 --bs 1 --no-torch --measure --overhead_test --unroll 2>&1 | tee $LOG_DIR/seq2seq.unroll.log
nvprof --profile-from-start off python3 seq2seq.py --platform V100 --bs 1 --no-torch --measure --overhead_test --unroll 2>&1 | tee ${LOG_DIR}/seq2seq.unroll.nvprof.log
python3 seq2seq.py --platform V100 --bs 1 --no-torch --measure --overhead_test --fix 2>&1 | tee $LOG_DIR/seq2seq.fix.log
nvprof --profile-from-start off python3 seq2seq.py --platform V100 --bs 1 --no-torch --measure --overhead_test --fix 2>&1 | tee $LOG_DIR/seq2seq.fix.nvprof.log
