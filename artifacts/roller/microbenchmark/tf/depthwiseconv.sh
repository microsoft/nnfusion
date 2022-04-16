mkdir -p ./logs/tf/depthwiseconv
CODE_DIR=./microbenchmark/tf/src
LOG_DIR=./logs/tf/depthwiseconv
REAPEAT_TIME=10000
python3 -u $CODE_DIR/depthwise.py 128 84 83 83 5 2 1 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/depthwiseconv0_128_84_83_83_5_2_1_SAME.log
python3 -u $CODE_DIR/depthwise.py 128 42 83 83 5 1 1 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/depthwiseconv1_128_42_83_83_5_1_1_SAME.log
python3 -u $CODE_DIR/depthwise.py 128 336 21 21 5 1 1 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/depthwiseconv2_128_336_21_21_5_1_1_SAME.log
python3 -u $CODE_DIR/depthwise.py 128 42 165 165 5 2 1 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/depthwiseconv3_128_42_165_165_5_2_1_SAME.log
python3 -u $CODE_DIR/depthwise.py 128 84 83 83 7 2 1 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/depthwiseconv4_128_84_83_83_7_2_1_SAME.log
python3 -u $CODE_DIR/depthwise.py 128 672 11 11 3 1 1 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/depthwiseconv5_128_672_11_11_3_1_1_SAME.log
python3 -u $CODE_DIR/depthwise.py 128 168 42 42 5 1 1 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/depthwiseconv6_128_168_42_42_5_1_1_SAME.log
python3 -u $CODE_DIR/depthwise.py 128 672 21 21 5 2 1 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/depthwiseconv7_128_672_21_21_5_2_1_SAME.log
python3 -u $CODE_DIR/depthwise.py 128 336 21 21 3 1 1 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/depthwiseconv8_128_336_21_21_3_1_1_SAME.log
python3 -u $CODE_DIR/depthwise.py 128 672 21 21 7 2 1 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/depthwiseconv9_128_672_21_21_7_2_1_SAME.log
python3 -u $CODE_DIR/depthwise.py 128 42 83 83 7 1 1 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/depthwiseconv10_128_42_83_83_7_1_1_SAME.log
python3 -u $CODE_DIR/depthwise.py 128 84 42 42 7 1 1 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/depthwiseconv11_128_84_42_42_7_1_1_SAME.log
python3 -u $CODE_DIR/depthwise.py 128 84 42 42 5 1 1 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/depthwiseconv12_128_84_42_42_5_1_1_SAME.log
python3 -u $CODE_DIR/depthwise.py 128 168 42 42 3 1 1 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/depthwiseconv13_128_168_42_42_3_1_1_SAME.log
python3 -u $CODE_DIR/depthwise.py 128 672 11 11 7 1 1 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/depthwiseconv14_128_672_11_11_7_1_1_SAME.log
python3 -u $CODE_DIR/depthwise.py 128 336 42 42 5 2 1 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/depthwiseconv15_128_336_42_42_5_2_1_SAME.log
python3 -u $CODE_DIR/depthwise.py 128 96 165 165 5 2 1 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/depthwiseconv16_128_96_165_165_5_2_1_SAME.log
python3 -u $CODE_DIR/depthwise.py 128 336 21 21 7 1 1 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/depthwiseconv17_128_336_21_21_7_1_1_SAME.log
python3 -u $CODE_DIR/depthwise.py 128 336 42 42 7 2 1 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/depthwiseconv18_128_336_42_42_7_2_1_SAME.log
python3 -u $CODE_DIR/depthwise.py 128 42 83 83 3 1 1 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/depthwiseconv19_128_42_83_83_3_1_1_SAME.log
python3 -u $CODE_DIR/depthwise.py 128 96 165 165 7 2 1 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/depthwiseconv20_128_96_165_165_7_2_1_SAME.log
python3 -u $CODE_DIR/depthwise.py 128 84 42 42 3 1 1 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/depthwiseconv21_128_84_42_42_3_1_1_SAME.log
python3 -u $CODE_DIR/depthwise.py 128 672 11 11 5 1 1 SAME $REAPEAT_TIME 2>&1 |tee $LOG_DIR/depthwiseconv22_128_672_11_11_5_1_1_SAME.log