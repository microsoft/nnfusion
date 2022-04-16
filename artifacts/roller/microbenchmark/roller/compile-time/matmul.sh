# training bs 128
LOG_DIR=logs/compile-time
CODE_DIR=compile-time

python3 $CODE_DIR/test_matmul_roller_compile_time.py 65536 2 1024 > $LOG_DIR/matmul_65536_2_1024.log 2>&1
python3 $CODE_DIR/test_matmul_roller_compile_time.py 128 4032 1000 > $LOG_DIR/matmul_128_4032_1000.log 2>&1
python3 $CODE_DIR/test_matmul_roller_compile_time.py 128 2048 1000 > $LOG_DIR/matmul_128_2048_1000.log 2>&1
python3 $CODE_DIR/test_matmul_roller_compile_time.py 65536 1024 4096 > $LOG_DIR/matmul_65536_1024_4096.log 2>&1
python3 $CODE_DIR/test_matmul_roller_compile_time.py 65536 1024 1024 > $LOG_DIR/matmul_65536_1024_1024.log 2>&1
python3 $CODE_DIR/test_matmul_roller_compile_time.py 65536 4096 1024 > $LOG_DIR/matmul_65536_4096_1024.log 2>&1
python3 $CODE_DIR/test_matmul_roller_compile_time.py 65536 30522 1024 > $LOG_DIR/matmul_65536_30522_1024.log 2>&1
