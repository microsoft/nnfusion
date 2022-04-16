# training bs 128
LOG_DIR=logs/ansor/scale-out/conv2d
CODE_DIR=microbenchmark/tvm/ansor

python3.7 -u $CODE_DIR/conv2d_tuning.py 8192 1024 14 14 2048 1 1 2 1 VALID 2>&1 |tee $LOG_DIR/ansor_conv_8192_1024_14_14_2048_1_1_2_1_VALID.log
python3.7 -u $CODE_DIR/conv2d_tuning.py 4096 1024 14 14 2048 1 1 2 1 VALID 2>&1 |tee $LOG_DIR/ansor_conv_4096_1024_14_14_2048_1_1_2_1_VALID.log
python3.7 -u $CODE_DIR/conv2d_tuning.py 2048 1024 14 14 2048 1 1 2 1 VALID 2>&1 |tee $LOG_DIR/ansor_conv_2048_1024_14_14_2048_1_1_2_1_VALID.log
python3.7 -u $CODE_DIR/conv2d_tuning.py 1024 1024 14 14 2048 1 1 2 1 VALID 2>&1 |tee $LOG_DIR/ansor_conv_1024_1024_14_14_2048_1_1_2_1_VALID.log
python3.7 -u $CODE_DIR/conv2d_tuning.py 512 1024 14 14 2048 1 1 2 1 VALID 2>&1 |tee $LOG_DIR/ansor_conv_512_1024_14_14_2048_1_1_2_1_VALID.log
python3.7 -u $CODE_DIR/conv2d_tuning.py 256 1024 14 14 2048 1 1 2 1 VALID 2>&1 |tee $LOG_DIR/ansor_conv_256_1024_14_14_2048_1_1_2_1_VALID.log
python3.7 -u $CODE_DIR/conv2d_tuning.py 64 1024 14 14 2048 1 1 2 1 VALID 2>&1 |tee $LOG_DIR/ansor_conv_64_1024_14_14_2048_1_1_2_1_VALID.log
python3.7 -u $CODE_DIR/conv2d_tuning.py 128 1024 14 14 2048 1 1 2 1 VALID 2>&1 |tee $LOG_DIR/ansor_conv_128_1024_14_14_2048_1_1_2_1_VALID.log
