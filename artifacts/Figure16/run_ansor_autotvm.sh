CURRENT_DIR=$PWD
SOURCE_DIR=$CURRENT_DIR/../microbenchmark/tvm
ANSOR_LOG_DIR=$SOURCE_DIR/ansor
AUTOTVM_LOG_DIR=$SOURCE_DIR/autotvm
OUTPUT_LOG_DIR=$CURRENT_DIR/logs
mkdir -p $OUTPUT_LOG_DIR

cd $SOURCE_DIR
# reproduce ansor results
bash script/small_op/bench0_matmul.sh ansor $ANSOR_LOG_DIR/matmul >> $OUTPUT_LOG_DIR/ansor_small_op.log 

# reproduce autotvm results
bash script/small_op/bench0_matmul.sh autotvm $AUTOTVM_LOG_DIR/matmul_nn >> $OUTPUT_LOG_DIR/autotvm_small_op.log
