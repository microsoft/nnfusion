CURRENT_DIR=$PWD
SOURCE_DIR=$CURRENT_DIR/../microbenchmark/tvm
ANSOR_LOG_DIR=$SOURCE_DIR/ansor
AUTOTVM_LOG_DIR=$SOURCE_DIR/autotvm
OUTPUT_LOG_DIR=$CURRENT_DIR/logs

mkdir -p $OUTPUT_LOG_DIR

mkdir -p $ANSOR_LOG_DIR/conv
mkdir -p $ANSOR_LOG_DIR/depthwise
mkdir -p $ANSOR_LOG_DIR/elementwise
mkdir -p $ANSOR_LOG_DIR/matmul
mkdir -p $ANSOR_LOG_DIR/pooling
mkdir -p $ANSOR_LOG_DIR/reduction

mkdir -p $AUTOTVM_LOG_DIR/conv
mkdir -p $AUTOTVM_LOG_DIR/depthwise
mkdir -p $AUTOTVM_LOG_DIR/elementwise
mkdir -p $AUTOTVM_LOG_DIR/matmul_nn
mkdir -p $AUTOTVM_LOG_DIR/pooling
mkdir -p $AUTOTVM_LOG_DIR/reduction

cd $SOURCE_DIR
# reproduce ansor results
bash script/op/bench0_conv2d.sh ansor $ANSOR_LOG_DIR/conv >> $OUTPUT_LOG_DIR/ansor_op.log 
bash script/op/bench0_depthwise_conv.sh ansor $ANSOR_LOG_DIR/depthwise >> $OUTPUT_LOG_DIR/ansor_op.log 
bash script/op/bench0_elementwise.sh ansor $ANSOR_LOG_DIR/elementwise >> $OUTPUT_LOG_DIR/ansor_op.log 
bash script/op/bench0_matmul.sh ansor $ANSOR_LOG_DIR/matmul >> $OUTPUT_LOG_DIR/ansor_op.log 
bash script/op/bench0_pooling.sh ansor $ANSOR_LOG_DIR/pooling >> $OUTPUT_LOG_DIR/ansor_op.log 
bash script/op/bench0_reduction.sh ansor $ANSOR_LOG_DIR/reduction >> $OUTPUT_LOG_DIR/ansor_op.log 

# reproduce autotvm results
bash script/op/bench0_conv2d.sh autotvm $AUTOTVM_LOG_DIR/conv >> $OUTPUT_LOG_DIR/autotvm_op.log
bash script/op/bench0_depthwise_conv.sh autotvm $AUTOTVM_LOG_DIR/depthwise >> $OUTPUT_LOG_DIR/autotvm_op.log
bash script/op/bench0_elementwise.sh autotvm $AUTOTVM_LOG_DIR/elementwise >> $OUTPUT_LOG_DIR/autotvm_op.log
bash script/op/bench0_matmul.sh autotvm $AUTOTVM_LOG_DIR/matmul_nn >> $OUTPUT_LOG_DIR/autotvm_op.log
bash script/op/bench0_pooling.sh autotvm $AUTOTVM_LOG_DIR/pooling >> $OUTPUT_LOG_DIR/autotvm_op.log
bash script/op/bench0_reduction.sh autotvm $AUTOTVM_LOG_DIR/reduction >> $OUTPUT_LOG_DIR/autotvm_op.log
