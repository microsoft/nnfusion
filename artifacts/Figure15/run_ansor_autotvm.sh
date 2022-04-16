CURRENT_DIR=$PWD
SOURCE_DIR=$CURRENT_DIR/../microbenchmark/tvm
LOG_DIR=$SOURCE_DIR/autotvm
OUTPUT_LOG_DIR=$CURRENT_DIR/logs
mkdir -p $OUTPUT_LOG_DIR

cd $SOURCE_DIR
# reproduce autotvm results
bash script/tensor_core/bench0_matmul_tensor_core.sh autotvm $LOG_DIR/tensor_core >> $OUTPUT_LOG_DIR/autotvm_tensor_core.log
