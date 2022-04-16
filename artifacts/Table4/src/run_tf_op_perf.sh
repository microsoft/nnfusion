# tvm backend: ROCM (tensorflow1.15)

CURRENT_DIR=$PWD
TF_DIR=$CURRENT_DIR/../../microbenchmark/tf
LOG_DIR=$CURRENT_DIR/../logs
TF_LOG_DIR_DEST=$LOG_DIR/tf/
TF_LOG_DIR_SRC=$TF_DIR/logs

# reproduce cudalib results and generate TF's log file
cd $TF_DIR
bash op_perf.sh

mv TF_LOG_DIR_SRC/* TF_LOG_DIR_DEST