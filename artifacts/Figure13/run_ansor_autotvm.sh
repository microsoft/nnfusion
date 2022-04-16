CURRENT_DIR=$PWD
SOURCE_DIR=$CURRENT_DIR/../microbenchmark/tvm
LOG_DIR=$SOURCE_DIR/scale_out
OUTPUT_LOG_DIR=$CURRENT_DIR/logs
mkdir -p $OUTPUT_LOG_DIR

cd $SOURCE_DIR
# reproduce ansor results
bash script/scale/bench0_conv2d_scale.sh ansor $LOG_DIR/conv_scale > $OUTPUT_LOG_DIR/ansor_conv_scale.log 

# reproduce autotvm results
bash script/scale/bench0_conv2d_scale.sh autotvm $LOG_DIR/conv_scale > $OUTPUT_LOG_DIR/autotvm_conv_scale.log
