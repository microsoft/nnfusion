# training bs 128
LOG_DIR=logs/sokoban/matmul
CODE_DIR=.

python3.7 -u $CODE_DIR/test_matmul.py 65536 2 1024 2>&1 |tee $LOG_DIR/matmul_65536_2_1024.log
python3.7 -u $CODE_DIR/test_matmul.py 128 4032 1000 2>&1 |tee $LOG_DIR/matmul_128_4032_1000.log
python3.7 -u $CODE_DIR/test_matmul.py 128 2048 1000 2>&1 |tee $LOG_DIR/matmul_128_2048_1000.log
python3.7 -u $CODE_DIR/test_matmul.py 65536 1024 4096 2>&1 |tee $LOG_DIR/matmul_65536_1024_4096.log
python3.7 -u $CODE_DIR/test_matmul.py 65536 1024 1024 2>&1 |tee $LOG_DIR/matmul_65536_1024_1024.log
python3.7 -u $CODE_DIR/test_matmul.py 65536 4096 1024 2>&1 |tee $LOG_DIR/matmul_65536_4096_1024.log
python3.7 -u $CODE_DIR/test_matmul.py 65536 30522 1024 2>&1 |tee $LOG_DIR/matmul_65536_30522_1024.log
