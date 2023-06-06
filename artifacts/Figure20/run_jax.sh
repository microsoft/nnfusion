#!/bin/bash

LOG_DIR=${ARTIFACT_ROOT}/reproduce_results/Figure20/jax
mkdir -p ${LOG_DIR}
cd ${ARTIFACT_ROOT}/baseline
cd lstm
python3 lstm_jax.py --platform MI100 --bs 1 2>&1 | tee ${LOG_DIR}/lstm.b1.log
cd ..
cd nasrnn
python3 nas_jax.py --platform MI100 --bs 1 2>&1 | tee ${LOG_DIR}/nasrnn.b1.log
cd ..
cd attention
python3 attention_jax.py --platform MI100 --bs 1 2>&1 | tee ${LOG_DIR}/attention.b1.log
cd ..
cd seq2seq
python3 seq2seq_jax.py --platform MI100 --bs 1 2>&1 | tee ${LOG_DIR}/seq2seq.b1.log
cd ..
cd blockdrop
python3 blockdrop_jax.py --platform MI100 --bs 1 --rand_weight 2>&1 | tee ${LOG_DIR}/blockdrop.b1.log
cd ..
cd skipnet
python3 skipnet_jax.py --platform MI100 --bs 1 --rand_weight 2>&1 | tee ${LOG_DIR}/skipnet.b1.log
cd ..
cd rae
python3 rae_jax.py --platform MI100 --bs 1 2>&1 | tee ${LOG_DIR}/rae.b1.log
cd ..