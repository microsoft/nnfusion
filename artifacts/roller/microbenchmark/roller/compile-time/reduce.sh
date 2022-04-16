# training bs 128
LOG_DIR=logs/compile-time
CODE_DIR=compile-time

python3 $CODE_DIR/test_reduce_roller.py 128 512 1024 1 > $LOG_DIR/reduce_128_512_1024_1.log
python3 $CODE_DIR/test_reduce_roller.py 65536 1024 1 > $LOG_DIR/reduce_65536_1024_1.log
python3 $CODE_DIR/test_reduce_roller.py 128 4032 11 11 2 > $LOG_DIR/reduce_128_4032_11_11_2.log
python3 $CODE_DIR/test_reduce_roller.py 128 2048 7 7 2 > $LOG_DIR/reduce_128_2048_7_7_2.log
