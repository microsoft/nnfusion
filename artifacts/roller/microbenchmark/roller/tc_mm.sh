# training bs 128
LOG_DIR=logs/roller/tc_mm
CODE_DIR=.

python3.7 -u $CODE_DIR/test_tc_matmul_roller.py 65536 1024 4096 2>&1 |tee $LOG_DIR/tc_matmul_65536_1024_4096.log
python3.7 -u $CODE_DIR/test_tc_matmul_roller.py 65536 1024 1024 2>&1 |tee $LOG_DIR/tc_matmul_65536_1024_1024.log
python3.7 -u $CODE_DIR/test_tc_matmul_roller.py 65536 4096 1024 2>&1 |tee $LOG_DIR/tc_matmul_65536_4096_1024.log
python3.7 -u $CODE_DIR/test_tc_matmul_roller.py 65536 16384 1024 2>&1 |tee $LOG_DIR/tc_matmul_65536_16384_1024.log
