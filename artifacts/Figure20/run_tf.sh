#!/bin/bash

LOG_DIR=${ARTIFACT_ROOT}/reproduce_results/Figure20/tf
mkdir -p ${LOG_DIR}
cd ${ARTIFACT_ROOT}/baseline
cd lstm
python3 lstm_tf.py --platform MI100 --bs 1 2>&1 | tee ${LOG_DIR}/lstm.b1.log
cd ..
cd nasrnn
python3 nas_tf.py --platform MI100 --bs 1 2>&1 | tee ${LOG_DIR}/nasrnn.b1.log
cd ..
cd attention
cp ${ARTIFACT_ROOT}/data/attention/attention.b1.tfgraph attention.b1.tfgraph
python3 attention_tf.py --platform MI100 --bs 1 2>&1 | tee ${LOG_DIR}/attention.b1.log
cd ..
cd seq2seq
cp ${ARTIFACT_ROOT}/data/seq2seq/seq2seq.b1.tfgraph seq2seq.b1.tfgraph
python3 seq2seq_tf.py --platform MI100 --bs 1 2>&1 | tee ${LOG_DIR}/seq2seq.b1.log
cd ..
cd blockdrop
mkdir -p onnx
cp ${ARTIFACT_ROOT}/data/blockdrop/blockdrop.b1.tfgraph blockdrop.b1.tfgraph
python3 blockdrop_tf.py --platform MI100 --bs 1 2>&1 | tee ${LOG_DIR}/blockdrop.b1.log
cd ..
cd skipnet
cp ${ARTIFACT_ROOT}/data/skipnet/skipnet.b1.tfgraph skipnet.b1.tfgraph
python3 skipnet_tf.py --platform MI100 --bs 1 2>&1 | tee ${LOG_DIR}/skipnet.b1.log
cd ..
