#!/bin/bash

LOG_DIR=${ARTIFACT_ROOT}/reproduce_results/Figure14/sys
mkdir -p ${LOG_DIR}
cd ${ARTIFACT_ROOT}/models
python3 lstm.py --platform V100 --bs 1 --no-torch --measure 2>&1 | tee $LOG_DIR/lstm.b1.log
python3 lstm.py --platform V100 --bs 64 --no-torch --measure 2>&1 | tee $LOG_DIR/lstm.b64.log
python3 nasrnn.py --platform V100 --bs 1 --no-torch --measure 2>&1 | tee $LOG_DIR/nasrnn.b1.log
python3 nasrnn.py --platform V100 --bs 64 --no-torch --measure 2>&1 | tee $LOG_DIR/nasrnn.b64.log
python3 seq2seq.py --platform V100 --bs 1 --no-torch --measure 2>&1 | tee $LOG_DIR/seq2seq.b1.log
python3 seq2seq.py --platform V100 --bs 64 --no-torch --measure 2>&1 | tee $LOG_DIR/seq2seq.b64.log
python3 attention.py --platform V100 --bs 1 --no-torch --measure 2>&1 | tee $LOG_DIR/attention.b1.log
python3 attention.py --platform V100 --bs 64 --no-torch --measure 2>&1 | tee $LOG_DIR/attention.b64.log
python3 blockdrop.py --platform V100 --bs 1 --no-torch --measure 2>&1 | tee $LOG_DIR/blockdrop.b1.log
python3 blockdrop.py --platform V100 --bs 64 --no-torch --measure 2>&1 | tee $LOG_DIR/blockdrop.b64.log
python3 skipnet.py --platform V100 --bs 1 --no-torch --measure 2>&1 | tee $LOG_DIR/skipnet.b1.log
python3 rae.py --platform V100 --bs 1 --no-torch --measure 2>&1 | tee $LOG_DIR/rae.b1.log
