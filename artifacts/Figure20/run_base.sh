#!/bin/bash

LOG_DIR=${ARTIFACT_ROOT}/reproduce_results/Figure20/base
mkdir -p ${LOG_DIR}
cd ${ARTIFACT_ROOT}/models

# build modified models
DIR_MAP=(
"manual_seq2seq/bs1.rocm/seq2seq_bs1_0-forward/nnfusion_rt/rocm_codegen:seq2seq/Constant.bs1.rocm.0"
"manual_attention/bs1.rocm/attention_bs1_0-forward/nnfusion_rt/rocm_codegen:attention/Constant.bs1.rocm.0"
"manual_attention/bs1.rocm/attention_bs1_1-forward/nnfusion_rt/rocm_codegen:attention/Constant.bs1.rocm.1"
)

for pair in ${DIR_MAP[@]}; do
    IFS=':' read -ra ADDR <<< "$pair"
    workdir=${ADDR[0]}
    datadir=${ADDR[1]}
    if [ ! -r $workdir/main_test ]; then
        echo "preparing $workdir"
        cd $workdir
        cp -r ${ARTIFACT_ROOT}/data/$datadir Constant
        cmake . && make -j
        cd ${ARTIFACT_ROOT}/models
    fi
done

# run tests
python3 lstm.py --platform MI100 --bs 1 --no-torch --disable-cf --measure 2>&1 | tee $LOG_DIR/lstm.b1.log
python3 nasrnn.py --platform MI100 --bs 1 --no-torch --disable-cf --measure 2>&1 | tee $LOG_DIR/nasrnn.b1.log
cd manual_attention/bs1.rocm && python3 run.py 2>&1 | tee $LOG_DIR/attention.b1.log && cd ../..
cd manual_seq2seq/bs1.rocm && python3 run.py 2>&1 | tee $LOG_DIR/seq2seq.b1.log && cd ../..
python3 blockdrop.py --platform MI100 --bs 1 --no-torch --disable-cf --measure 2>&1 | tee $LOG_DIR/blockdrop.b1.log
python3 skipnet.py --platform MI100 --bs 1 --no-torch --disable-cf --measure 2>&1 | tee $LOG_DIR/skipnet.b1.log
python3 rae.py --platform MI100 --bs 1 --no-torch --disable-cf --measure 2>&1 | tee $LOG_DIR/rae.b1.log
