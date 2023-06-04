#!/bin/bash

export ARTIFACT_ROOT=$(pwd)
mkdir -p ${ARTIFACT_ROOT}/reproduce_results
date > ${ARTIFACT_ROOT}/reproduce_results/start_time.txt
cd Figure14 && ./run.sh && cd -
date > ${ARTIFACT_ROOT}/reproduce_results/Figure14/finish_time.txt
cd Figure15 && ./run.sh && cd -
date > ${ARTIFACT_ROOT}/reproduce_results/Figure15/finish_time.txt
cd Figure16 && ./run.sh && cd -
date > ${ARTIFACT_ROOT}/reproduce_results/Figure16/finish_time.txt
cd Figure17 && ./run.sh && cd -
date > ${ARTIFACT_ROOT}/reproduce_results/Figure17/finish_time.txt
cd Figure18 && ./run.sh && cd -
date > ${ARTIFACT_ROOT}/reproduce_results/Figure18/finish_time.txt
cd Figure19 && ./run.sh && cd -
date > ${ARTIFACT_ROOT}/reproduce_results/Figure19/finish_time.txt
