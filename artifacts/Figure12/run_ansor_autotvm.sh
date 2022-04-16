CURRENT_DIR=$PWD
SOURCE_DIR=$CURRENT_DIR/../microbenchmark/tvm
LOG_DIR=$SOURCE_DIR/scale_out
OUTPUT_LOG_DIR=$CURRENT_DIR/logs
mkdir -p $OUTPUT_LOG_DIR

cd $SOURCE_DIR
# reproduce ansor results
bash script/scale/bench0_matmul_bert_scale.sh ansor $LOG_DIR/matmul_bert_scale >> $OUTPUT_LOG_DIR/ansor_matmul_scale.log 

# reproduce autotvm results
bash script/scale/bench0_matmul_bert_scale.sh autotvm $LOG_DIR/matmul_bert_scale >> $OUTPUT_LOG_DIR/autotvm_matmul_scale.log
