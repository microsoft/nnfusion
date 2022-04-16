# tvm backend: CUDA

CURRENT_DIR=$PWD
ROLLER_DIR=$CURRENT_DIR/../../roller
LOG_DIR=$CURRENT_DIR/../logs
ROLLER_LOG_DIR_DEST=$LOG_DIR/roller/wo_storage_align/
ROLLER_LOG_DIR_SRC=$ROLLER_DIR/logs_v2_rocm/roller/wo_storage_align

# reproduce results and generate roller's log file
cd $ROLLER_DIR
bash script_v2_rocm/conv.sh
bash script_v2_rocm/matmul.sh
bash script_v2_rocm/depthwiseconv.sh
bash script_v2_rocm/reduce.sh
bash script_v2_rocm/elementwise.sh
bash script_v2_rocm/pooling.sh

mv $ROLLER_LOG_DIR_SRC/* $ROLLER_LOG_DIR_DEST