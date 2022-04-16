# training bs 128
LOG_DIR=logs/roller/scale-out/relu
CODE_DIR=.

python3.7 -u $CODE_DIR/test_elementwise_roller.py relu 1024 1024 2>&1 |tee $LOG_DIR/relu_1024_1024.log
python3.7 -u $CODE_DIR/test_elementwise_roller.py relu 2048 1024 2>&1 |tee $LOG_DIR/relu_2048_1024.log
python3.7 -u $CODE_DIR/test_elementwise_roller.py relu 4096 1024 2>&1 |tee $LOG_DIR/relu_4096_1024.log
python3.7 -u $CODE_DIR/test_elementwise_roller.py relu 8192 1024 2>&1 |tee $LOG_DIR/relu_8192_1024.log
python3.7 -u $CODE_DIR/test_elementwise_roller.py relu 16384 1024 2>&1 |tee $LOG_DIR/relu_16384_1024.log
python3.7 -u $CODE_DIR/test_elementwise_roller.py relu 32768 1024 2>&1 |tee $LOG_DIR/relu_32768_1024.log
python3.7 -u $CODE_DIR/test_elementwise_roller.py relu 65536 1024 2>&1 |tee $LOG_DIR/relu_65536_1024.log
