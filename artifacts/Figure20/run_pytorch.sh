#!/bin/bash

LOG_DIR=${ARTIFACT_ROOT}/reproduce_results/Figure20/pytorch
mkdir -p ${LOG_DIR}
cd ${ARTIFACT_ROOT}/baseline
cd lstm
python3 lstm_pytorch.py --platform MI100 --bs 1 2>&1 | tee ${LOG_DIR}/lstm.b1.log
cd ..
cd nasrnn
python3 nas_pytorch.py --platform MI100 --bs 1 2>&1 | tee ${LOG_DIR}/nasrnn.b1.log
cd ..
cd attention
python3 attention_pytorch.py --platform MI100 --bs 1 2>&1 | tee ${LOG_DIR}/attention.b1.log
cd ..
cd seq2seq
python3 seq2seq_pytorch.py --platform MI100 --bs 1 2>&1 | tee ${LOG_DIR}/seq2seq.b1.log
cd ..
cd blockdrop
python3 blockdrop_pytorch.py --platform MI100 --bs 1 2>&1 | tee ${LOG_DIR}/blockdrop.b1.log
cd ..
cd skipnet
python3 skipnet_pytorch.py --platform MI100 --bs 1 2>&1 | tee ${LOG_DIR}/skipnet.b1.log
cd ..
cd rae
python3 rae_pytorch.py --platform MI100 --bs 1 2>&1 | tee ${LOG_DIR}/rae.b1.log
cd ..