CURRENT_DIR=$PWD
SOURCE_DIR=$CURRENT_DIR/../microbenchmark/tvm
ANSOR_LOG_DIR=$SOURCE_DIR/ansor
AUTOTVM_LOG_DIR=$SOURCE_DIR/autotvm
OUTPUT_LOG_DIR=$CURRENT_DIR/logs
mkdir -p $OUTPUT_LOG_DIR

cd $SOURCE_DIR
# reproduce ansor results
bash script/irregular_conv/bench0_conv2d.sh ansor $ANSOR_LOG_DIR/conv >> $OUTPUT_LOG_DIR/ansor_irregular_conv.log 

# reproduce autotvm results
bash script/irregular_conv/bench0_conv2d.sh autotvm $AUTOTVM_LOG_DIR/conv >> $OUTPUT_LOG_DIR/autotvm_irregular_conv.log
