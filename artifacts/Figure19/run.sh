#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate controlflow

cd ${ARTIFACT_ROOT}/models

mkdir -p ${ARTIFACT_ROOT}/reproduce_results/Figure19
cp -r ${ARTIFACT_ROOT}/reproduce_results/Figure14/base ${ARTIFACT_ROOT}/reproduce_results/Figure19
cp -r ${ARTIFACT_ROOT}/reproduce_results/Figure14/sys ${ARTIFACT_ROOT}/reproduce_results/Figure19

LOG_DIR=${ARTIFACT_ROOT}/reproduce_results/Figure19/schedule
mkdir -p ${LOG_DIR}
python3 lstm.py --platform V100 --bs 1 --no-torch --measure --enable-breakdown 2>&1 | tee $LOG_DIR/lstm.b1.log
cp ${ARTIFACT_ROOT}/reproduce_results/Figure14/sys/nasrnn.b1.log ${LOG_DIR}
python3 attention.py --platform V100 --bs 1 --no-torch --measure --enable-breakdown 2>&1 | tee $LOG_DIR/attention.b1.log
cp ${ARTIFACT_ROOT}/reproduce_results/Figure14/sys/seq2seq.b1.log ${LOG_DIR}
python3 blockdrop.py --platform V100 --bs 1 --no-torch --measure --enable-breakdown 2>&1 | tee $LOG_DIR/blockdrop.b1.log
python3 skipnet.py --platform V100 --bs 1 --no-torch --measure --enable-breakdown 2>&1 | tee $LOG_DIR/skipnet.b1.log
python3 rae.py --platform V100 --bs 1 --no-torch --measure --enable-breakdown --opt=1 2>&1 | tee $LOG_DIR/rae.opt1.b1.log
python3 rae.py --platform V100 --bs 1 --no-torch --measure --enable-breakdown --opt=2 2>&1 | tee $LOG_DIR/rae.opt2.b1.log
python3 rae.py --platform V100 --bs 1 --no-torch --measure --enable-breakdown --opt=3 2>&1 | tee $LOG_DIR/rae.opt3.b1.log
python3 rae.py --platform V100 --bs 1 --no-torch --measure --enable-breakdown --opt=4 2>&1 | tee $LOG_DIR/rae.opt4.b1.log

conda deactivate
