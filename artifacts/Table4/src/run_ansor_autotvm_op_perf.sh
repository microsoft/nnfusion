# tvm backend: ROCM

CURRENT_DIR=$PWD
ANSOR_LOG_DIR=$CURRENT_DIR../tvm/ansor
AUTOTVM_LOG_DIR=$CURRENT_DIR../tvm/autotvm

# reproduce ansor results
bash script/bench0_conv2d.sh ansor 
mv *.log ANSOR_LOG_DIR/conv
rm ansor_*

bash script/bench0_depthwise_conv.sh ansor 
mv *.log ANSOR_LOG_DIR/depthwise
rm ansor_*

bash script/bench0_elementwise.sh ansor 
mv *.log ANSOR_LOG_DIR/elementwise
rm ansor_*

bash script/bench0_matmul.sh ansor 
mv *.log ANSOR_LOG_DIR/matmul
rm ansor_*

bash script/bench0_pooling.sh ansor 
mv *.log ANSOR_LOG_DIR/pooling
rm ansor_*

bash script/bench0_reduction.sh ansor 
mv *.log ANSOR_LOG_DIR/reduction
rm ansor_*

# reproduce autotvm results
bash script/bench0_conv2d.sh autotvm 
mv *.log AUTOTVM_LOG_DIR/conv
rm conv2d*

bash script/bench0_depthwise_conv.sh autotvm 
mv *.log AUTOTVM_LOG_DIR/depthwise
rm depthwise*

bash script/bench0_elementwise.sh autotvm >>elementwise.log
mv *.log AUTOTVM_LOG_DIR/elementwise
rm elementwise*

bash script/bench0_matmul.sh autotvm 
mv *.log AUTOTVM_LOG_DIR/matmul
rm matmul*

bash script/bench0_pooling.sh autotvm >>pooling.log
mv *.log AUTOTVM_LOG_DIR/pooling
rm pooling*

bash script/bench0_reduction.sh autotvm >>reduction.log
mv *.log AUTOTVM_LOG_DIR/reduction
rm reduction*
