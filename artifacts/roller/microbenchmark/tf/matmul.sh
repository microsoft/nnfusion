mkdir -p ./logs/tf/matmul
CODE_DIR=./microbenchmark/tf/src
LOG_DIR=./logs/tf/matmul
REAPEAT_TIME=10000
python3 -u $CODE_DIR/matmul.py 65536 2 1024 $REAPEAT_TIME 2>&1 | tee $LOG_DIR/matmul0_65536_1024_2.log
python3 -u $CODE_DIR/matmul.py 128 4032 1000 $REAPEAT_TIME 2>&1 | tee $LOG_DIR/matmul1_128_1000_4032.log
python3 -u $CODE_DIR/matmul.py 128 2048 1000 $REAPEAT_TIME 2>&1 | tee $LOG_DIR/matmul2_128_1000_2048.log
python3 -u $CODE_DIR/matmul.py 65536 1024 4096 $REAPEAT_TIME 2>&1 | tee $LOG_DIR/matmul3_65536_4096_1024.log
python3 -u $CODE_DIR/matmul.py 65536 1024 1024 $REAPEAT_TIME 2>&1 | tee $LOG_DIR/matmul4_65536_1024_1024.log
python3 -u $CODE_DIR/matmul.py 65536 4096 1024 $REAPEAT_TIME 2>&1 | tee $LOG_DIR/matmul5_65536_1024_4096.log
python3 -u $CODE_DIR/matmul.py 65536 30522 1024 $REAPEAT_TIME 2>&1 | tee $LOG_DIR/matmul6_65536_1024_30522.log