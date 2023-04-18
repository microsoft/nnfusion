#!/bin/bash

LOG_DIR=${ARTIFACT_ROOT}/reproduce_results/Figure17/grinder
mkdir -p ${LOG_DIR}
cd ${ARTIFACT_ROOT}/../models

rates=(0 25 50 75 100)

python3 resnet18.py --platform V100 --bs 1 --no-torch --measure 2>&1 | tee ${LOG_DIR}/blockdrop.noskip.log

for rate in ${rates[*]}; do
    python3 blockdrop.py --bs 1 --no-torch --overhead_test --unroll --measure --rate $rate 2>&1 | tee ${LOG_DIR}/blockdrop.${rate}.unroll.log
    python3 blockdrop.py --bs 1 --no-torch --overhead_test --fix --measure --rate $rate 2>&1 | tee ${LOG_DIR}/blockdrop.${rate}.fix.log
done

python3 resnet101.py --platform V100 --bs 1 --no-torch --measure 2>&1 | tee ${LOG_DIR}/skipnet.noskip.log

for rate in ${rates[*]}; do
    python3 skipnet.py --bs 1 --no-torch --overhead_test --unroll --measure --rate $rate 2>&1 | tee ${LOG_DIR}/skipnet.${rate}.unroll.log
    python3 skipnet.py --bs 1 --no-torch --overhead_test --fix --measure --rate $rate 2>&1 | tee ${LOG_DIR}/skipnet.${rate}.fix.log
done
cd ..
