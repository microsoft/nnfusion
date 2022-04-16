# run code and generate logs go to src/ 

ROLLER_LOGS_DIR=./logs/roller/wo_storage_align/
TF_LOGS_DIR=./logs/tf/
AUTOTVM_DIR=./tvm/autotvm
ANSOR_DIR=./tvm/ansor
AUTOTVM_LOG_NAME=$AUTOTVM_DIR/autotvm_op_perf.txt
ANSOR_LOG_NAME=$ANSOR_DIR/ansor_op_perf.txt
CURRENT_DIR=$PWD

# parse autotvm log

# cd $AUTOTVM_DIR
# bash parse_op_log.sh

# parse ansor log

# cd $ANSOR_DIR
# bash parse_op_log.sh

# parse all logs and print the Table4

cd $CURRENT_DIR
python3 -u process_table4.py $ROLLER_LOGS_DIR $TF_LOGS_DIR $AUTOTVM_LOG_NAME $ANSOR_LOG_NAME