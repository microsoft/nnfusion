# training bs 128
LOG_DIR=logs/roller/scale-out/conv2d
CODE_DIR=.

python3.7 -u $CODE_DIR/test_implicit_gemm_padding_roller.py 64 1024 14 14 2048 1 1 2 1 VALID 2>&1 |tee $LOG_DIR/implicit_gemm_padding_roller_64_1024_14_14_2048_1_1_2_1_VALID.log
python3.7 -u $CODE_DIR/test_implicit_gemm_padding_roller.py 128 1024 14 14 2048 1 1 2 1 VALID 2>&1 |tee $LOG_DIR/implicit_gemm_padding_roller_128_1024_14_14_2048_1_1_2_1_VALID.log
python3.7 -u $CODE_DIR/test_implicit_gemm_padding_roller.py 256 1024 14 14 2048 1 1 2 1 VALID 2>&1 |tee $LOG_DIR/implicit_gemm_padding_roller_256_1024_14_14_2048_1_1_2_1_VALID.log
python3.7 -u $CODE_DIR/test_implicit_gemm_padding_roller.py 512 1024 14 14 2048 1 1 2 1 VALID 2>&1 |tee $LOG_DIR/implicit_gemm_padding_roller_512_1024_14_14_2048_1_1_2_1_VALID.log
python3.7 -u $CODE_DIR/test_implicit_gemm_padding_roller.py 1024 1024 14 14 2048 1 1 2 1 VALID 2>&1 |tee $LOG_DIR/implicit_gemm_padding_roller_1024_1024_14_14_2048_1_1_2_1_VALID.log
python3.7 -u $CODE_DIR/test_implicit_gemm_padding_roller.py 2048 1024 14 14 2048 1 1 2 1 VALID 2>&1 |tee $LOG_DIR/implicit_gemm_padding_roller_2048_1024_14_14_2048_1_1_2_1_VALID.log
python3.7 -u $CODE_DIR/test_implicit_gemm_padding_roller.py 4096 1024 14 14 2048 1 1 2 1 VALID 2>&1 |tee $LOG_DIR/implicit_gemm_padding_roller_4096_1024_14_14_2048_1_1_2_1_VALID.log
python3.7 -u $CODE_DIR/test_implicit_gemm_padding_roller.py 8192 1024 14 14 2048 1 1 2 1 VALID 2>&1 |tee $LOG_DIR/implicit_gemm_padding_roller_8192_1024_14_14_2048_1_1_2_1_VALID.log
