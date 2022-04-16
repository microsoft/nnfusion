# training bs 128 and 1
LOG_DIR=logs/sokoban/transpose
CODE_DIR=.

python3.7 -u $CODE_DIR/test_transpose.py 128 512 16 64 0 2 1 3 2>&1 |tee $LOG_DIR/transpose_128_512_16_64_0_2_1_3.log
python3.7 -u $CODE_DIR/test_transpose.py 128 16 512 64 0 2 1 3 2>&1 |tee $LOG_DIR/transpose_128_16_512_64_0_2_1_3.log
python3.7 -u $CODE_DIR/test_transpose.py 128 331 331 3 0 3 2 1 2>&1 |tee $LOG_DIR/transpose_128_331_331_3_0_3_2_1.log
python3.7 -u $CODE_DIR/test_transpose.py 128 331 331 3 0 3 1 2 2>&1 |tee $LOG_DIR/transpose_128_331_331_3_0_3_1_2.log

python3.7 -u $CODE_DIR/test_transpose.py 1 512 16 64 0 2 1 3 2>&1 |tee $LOG_DIR/transpose_1_512_16_64_0_2_1_3.log
python3.7 -u $CODE_DIR/test_transpose.py 1 16 512 64 0 2 1 3 2>&1 |tee $LOG_DIR/transpose_1_16_512_64_0_2_1_3.log
python3.7 -u $CODE_DIR/test_transpose.py 1 331 331 3 0 3 2 1 2>&1 |tee $LOG_DIR/transpose_1_331_331_3_0_3_2_1.log
python3.7 -u $CODE_DIR/test_transpose.py 1 331 331 3 0 3 1 2 2>&1 |tee $LOG_DIR/transpose_1_331_331_3_0_3_1_2.log