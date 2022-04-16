LOG_DIR=logs/roller/scale-out/matmul
CODE_DIR=.

python3.7 -u $CODE_DIR/test_matmul_roller.py 512 1024 4096 2>&1 |tee $LOG_DIR/matmul_512_1024_4096.log
python3.7 -u $CODE_DIR/test_matmul_roller.py 1024 1024 4096 2>&1 |tee $LOG_DIR/matmul_1024_1024_4096.log
python3.7 -u $CODE_DIR/test_matmul_roller.py 2048 1024 4096 2>&1 |tee $LOG_DIR/matmul_2048_1024_4096.log
python3.7 -u $CODE_DIR/test_matmul_roller.py 4096 1024 4096 2>&1 |tee $LOG_DIR/matmul_4096_1024_4096.log
python3.7 -u $CODE_DIR/test_matmul_roller.py 8192 1024 4096 2>&1 |tee $LOG_DIR/matmul_8192_1024_4096.log
python3.7 -u $CODE_DIR/test_matmul_roller.py 16384 1024 4096 2>&1 |tee $LOG_DIR/matmul_16384_1024_4096.log
