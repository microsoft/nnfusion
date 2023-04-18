#!/bin/bash

LOG_DIR=${ARTIFACT_ROOT}/reproduce_results/Figure17/tf
mkdir -p ${LOG_DIR}
cd ${ARTIFACT_ROOT}/baseline

rates=(0 25 50 75 100)

cd resnet18
python3 resnet18_tf.py --bs=1 --platform=V100 2>&1 | tee ${LOG_DIR}/blockdrop.noskip.log
cd ..

cd blockdrop
for rate in ${rates[*]}; do
    python3 blockdrop_tf.py --bs 1 --platform V100 --overhead_test --unroll --rate $rate 2>&1 | tee ${LOG_DIR}/blockdrop.${rate}.unroll.log
    python3 blockdrop_tf.py --bs 1 --platform V100 --overhead_test --fix --rate $rate 2>&1 | tee ${LOG_DIR}/blockdrop.${rate}.fix.log
done
cd ..

cd resnet101
python3 resnet101_tf.py --bs=1 --platform=V100 2>&1 | tee ${LOG_DIR}/blockdrop.noskip.log
cd ..

cd skipnet
for rate in ${rates[*]}; do
    python3 skipnet_tf.py --bs 1 --platform V100 --overhead_test --unroll --rate $rate 2>&1 | tee ${LOG_DIR}/skipnet.${rate}.unroll.log
    python3 skipnet_tf.py --bs 1 --platform V100 --overhead_test --fix --rate $rate 2>&1 | tee ${LOG_DIR}/skipnet.${rate}.fix.log
done
cd ..
