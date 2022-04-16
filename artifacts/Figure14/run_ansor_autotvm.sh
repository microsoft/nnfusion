CURRENT_DIR=$PWD
SOURCE_DIR=$CURRENT_DIR/../microbenchmark/tvm
LOG_DIR=$SOURCE_DIR/scale_out
OUTPUT_LOG_DIR=$CURRENT_DIR/logs
mkdir -p $OUTPUT_LOG_DIR

cd $SOURCE_DIR
# get best ansor result
bash script/scale/bench0_matmul_bert_scale.sh ansor $LOG_DIR/matmul_bert_scale >> $OUTPUT_LOG_DIR/ansor_matmul_scale.log 
bash script/scale/bench0_conv2d_scale.sh ansor $LOG_DIR/conv_scale >> $OUTPUT_LOG_DIR/ansor_conv_scale.log 

# get best autotvm result
bash script/scale/bench0_matmul_bert_scale.sh autotvm $LOG_DIR/matmul_bert_scale >> $OUTPUT_LOG_DIR/autotvm_matmul_scale.log
bash script/scale/bench0_conv2d_scale.sh autotvm $LOG_DIR/conv_scale >> $OUTPUT_LOG_DIR/autotvm_conv_scale.log
