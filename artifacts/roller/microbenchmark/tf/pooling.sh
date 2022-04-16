mkdir -p ./logs/tf/pooling
CODE_DIR=./microbenchmark/tf/src
LOG_DIR=./logs/tf/pooling
REAPEAT_TIME=10000
python3 -u $CODE_DIR/pool.py 128 168 83 83 1 2 VALID $REAPEAT_TIME 2>&1 |tee $LOG_DIR/pooling0_avg_128_168_83_83_1_2_VALID.log
python3 -u $CODE_DIR/pool.py 128 672 21 21 3 2 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/pooling1_avg_128_672_21_21_3_2_SAME.log
python3 -u $CODE_DIR/pool.py 128 42 83 83 3 1 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/pooling2_avg_128_42_83_83_3_1_SAME.log
python3 -u $CODE_DIR/pool.py 128 1008 42 42 1 2 VALID $REAPEAT_TIME 2>&1 |tee $LOG_DIR/pooling3_avg_128_1008_42_42_1_2_VALID.log
python3 -u $CODE_DIR/pool.py 128 336 42 42 3 2 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/pooling4_avg_128_336_42_42_3_2_SAME.log
python3 -u $CODE_DIR/pool.py 128 84 83 83 3 2 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/pooling5_avg_128_84_83_83_3_2_SAME.log
python3 -u $CODE_DIR/pool.py 128 672 11 11 3 1 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/pooling6_avg_128_672_11_11_3_1_SAME.log
python3 -u $CODE_DIR/pool.py 128 96 165 165 1 2 VALID $REAPEAT_TIME 2>&1 |tee $LOG_DIR/pooling7_avg_128_96_165_165_1_2_VALID.log
python3 -u $CODE_DIR/pool.py 128 2016 21 21 1 2 VALID $REAPEAT_TIME 2>&1 |tee $LOG_DIR/pooling8_avg_128_2016_21_21_1_2_VALID.log
python3 -u $CODE_DIR/pool.py 128 42 165 165 3 2 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/pooling9_avg_128_42_165_165_3_2_SAME.log
python3 -u $CODE_DIR/pool.py 128 84 42 42 3 1 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/pooling10_avg_128_84_42_42_3_1_SAME.log
python3 -u $CODE_DIR/pool.py 128 336 21 21 3 1 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/pooling11_avg_128_336_21_21_3_1_SAME.log
python3 -u $CODE_DIR/pool.py 128 168 42 42 3 1 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/pooling12_avg_128_168_42_42_3_1_SAME.log