mkdir -p ./logs/tc_matmul
CODE_DIR=./src
LOG_DIR=./logs/tc_matmul
REPEAT_TIME=100
python3 -u $CODE_DIR/tc_matmul.py 65536 1024 4096 $REPEAT_TIME 2>&1 | tee $LOG_DIR/tc_mm0_65536_4096_1024.log
python3 -u $CODE_DIR/tc_matmul.py 65536 1024 1024 $REPEAT_TIME 2>&1 | tee $LOG_DIR/tc_mm1_65536_1024_1024.log
python3 -u $CODE_DIR/tc_matmul.py 65536 4096 1024 $REPEAT_TIME 2>&1 | tee $LOG_DIR/tc_mm2_65536_1024_4096.log
python3 -u $CODE_DIR/tc_matmul.py 65536 16384 1024 $REPEAT_TIME 2>&1 | tee $LOG_DIR/tc_mm3_65536_1024_16384.log
