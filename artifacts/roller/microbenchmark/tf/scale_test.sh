mkdir -p ./logs/tf/scale/conv
CODE_DIR=./microbenchmark/tf/src
LOG_DIR=./logs/tf/scale/conv
REAPEAT_TIME=10000
python3 -u $CODE_DIR/conv.py 128 1024 14 14 2048 1 1 2 1 VALID $REAPEAT_TIME 2>&1 |tee $LOG_DIR/conv0_128_1024_14_14_2048_1_1_2_1_VALID.log
python3 -u $CODE_DIR/conv.py 256 1024 14 14 2048 1 1 2 1 VALID $REAPEAT_TIME 2>&1 |tee $LOG_DIR/conv1_256_1024_14_14_2048_1_1_2_1_VALID.log
python3 -u $CODE_DIR/conv.py 512 1024 14 14 2048 1 1 2 1 VALID $REAPEAT_TIME 2>&1 |tee $LOG_DIR/conv2_512_1024_14_14_2048_1_1_2_1_VALID.log
python3 -u $CODE_DIR/conv.py 1024 1024 14 14 2048 1 1 2 1 VALID $REAPEAT_TIME 2>&1 |tee $LOG_DIR/conv3_1024_1024_14_14_2048_1_1_2_1_VALID.log
python3 -u $CODE_DIR/conv.py 2048 1024 14 14 2048 1 1 2 1 VALID $REAPEAT_TIME 2>&1 |tee $LOG_DIR/conv4_2048_1024_14_14_2048_1_1_2_1_VALID.log
python3 -u $CODE_DIR/conv.py 4096 1024 14 14 2048 1 1 2 1 VALID $REAPEAT_TIME 2>&1 |tee $LOG_DIR/conv5_4096_1024_14_14_2048_1_1_2_1_VALID.log
python3 -u $CODE_DIR/conv.py 8192 1024 14 14 2048 1 1 2 1 VALID $REAPEAT_TIME 2>&1 |tee $LOG_DIR/conv6_8192_1024_14_14_2048_1_1_2_1_VALID.log

mkdir -p ./logs/tf/scale/matmul
CODE_DIR=./microbenchmark/tf/src
LOG_DIR=./logs/tf/scale/matmul
REAPEAT_TIME=10000
python3 -u $CODE_DIR/matmul.py 512 4096 1024 $REAPEAT_TIME 2>&1 | tee $LOG_DIR/matmul0_512_1024_4096.log
python3 -u $CODE_DIR/matmul.py 1024 4096 1024 $REAPEAT_TIME 2>&1 | tee $LOG_DIR/matmul1_1024_1024_4096.log
python3 -u $CODE_DIR/matmul.py 2048 4096 1024 $REAPEAT_TIME 2>&1 | tee $LOG_DIR/matmul2_2048_1024_4096.log
python3 -u $CODE_DIR/matmul.py 4096 4096 1024 $REAPEAT_TIME 2>&1 | tee $LOG_DIR/matmul3_4096_1024_4096.log
python3 -u $CODE_DIR/matmul.py 8192 4096 1024 $REAPEAT_TIME 2>&1 | tee $LOG_DIR/matmul4_8192_1024_4096.log
python3 -u $CODE_DIR/matmul.py 16384 4096 1024 $REAPEAT_TIME 2>&1 | tee $LOG_DIR/matmul5_16384_1024_4096.log