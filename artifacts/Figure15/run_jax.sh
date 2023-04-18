#!/bin/bash

LOG_DIR=${ARTIFACT_ROOT}/reproduce_results/Figure15/jax
mkdir -p ${LOG_DIR}
cd ${ARTIFACT_ROOT}/../baseline

cd lstm
python3 lstm_jax.py --bs 1 --platform V100 --overhead_test --unroll 2>&1 | tee ${LOG_DIR}/lstm.unroll.log
nvprof --profile-from-start off python3 lstm_jax.py --bs 1 --platform V100 --overhead_test --unroll 2>&1 | tee ${LOG_DIR}/lstm.unroll.nvprof.log
python3 lstm_jax.py --bs 1 --platform V100 --overhead_test --fix 2>&1 | tee ${LOG_DIR}/lstm.fix.log
nvprof --profile-from-start off python3 lstm_jax.py --bs 1 --platform V100 --overhead_test --fix 2>&1 | tee ${LOG_DIR}/lstm.fix.nvprof.log
cd ..
cd nasrnn
python3 nas_jax.py --bs 1 --platform V100 --overhead_test --unroll 2>&1 | tee ${LOG_DIR}/nasrnn.unroll.log
nvprof --profile-from-start off python3 nas_jax.py --bs 1 --platform V100 --overhead_test --unroll 2>&1 | tee ${LOG_DIR}/nasrnn.unroll.nvprof.log
python3 nas_jax.py --bs 1 --platform V100 --overhead_test --fix 2>&1 | tee ${LOG_DIR}/nasrnn.fix.log
nvprof --profile-from-start off python3 nas_jax.py --bs 1 --platform V100 --overhead_test --fix 2>&1 | tee ${LOG_DIR}/nasrnn.fix.nvprof.log
cd ..
cd attention
python3 attention_jax.py --bs 1 --platform V100 --overhead_test --unroll 2>&1 | tee ${LOG_DIR}/attention.unroll.log
nvprof --profile-from-start off python3 attention_jax.py --bs 1 --platform V100 --overhead_test --unroll 2>&1 | tee ${LOG_DIR}/attention.unroll.nvprof.log
python3 attention_jax.py --bs 1 --platform V100 --overhead_test --fix 2>&1 | tee ${LOG_DIR}/attention.fix.log
nvprof --profile-from-start off python3 attention_jax.py --bs 1 --platform V100 --overhead_test --fix 2>&1 | tee ${LOG_DIR}/attention.fix.nvprof.log
cd ..
cd seq2seq
python3 seq2seq_jax.py --bs 1 --platform V100 --overhead_test --unroll 2>&1 | tee ${LOG_DIR}/seq2seq.unroll.log
nvprof --profile-from-start off python3 seq2seq_jax.py --bs 1 --platform V100 --overhead_test --unroll 2>&1 | tee ${LOG_DIR}/seq2seq.unroll.nvprof.log
python3 seq2seq_jax.py --bs 1 --platform V100 --overhead_test --fix 2>&1 | tee ${LOG_DIR}/seq2seq.fix.log
nvprof --profile-from-start off python3 seq2seq_jax.py --bs 1 --platform V100 --overhead_test --fix 2>&1 | tee ${LOG_DIR}/seq2seq.fix.nvprof.log
cd ..
cd ..