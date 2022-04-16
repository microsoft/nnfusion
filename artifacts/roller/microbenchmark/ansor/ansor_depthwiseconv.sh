# training bs 128
LOG_DIR=logs/ansor/depthwiseconv
CODE_DIR=microbenchmark/tvm/ansor

python3.7 -u $CODE_DIR/depthwise_tuning.py 128 84 83 83 5 5 2 1 SAME 2>&1 |tee $LOG_DIR/ansor_depthwise_128_84_83_83_5_5_2_1_SAME.log
python3.7 -u $CODE_DIR/depthwise_tuning.py 128 42 83 83 5 5 1 1 SAME 2>&1 |tee $LOG_DIR/ansor_depthwise_128_42_83_83_5_5_1_1_SAME.log
python3.7 -u $CODE_DIR/depthwise_tuning.py 128 336 21 21 5 5 1 1 SAME 2>&1 |tee $LOG_DIR/ansor_depthwise_128_336_21_21_5_5_1_1_SAME.log
python3.7 -u $CODE_DIR/depthwise_tuning.py 128 42 165 165 5 5 2 1 SAME 2>&1 |tee $LOG_DIR/ansor_depthwise_128_42_165_165_5_5_2_1_SAME.log
python3.7 -u $CODE_DIR/depthwise_tuning.py 128 84 83 83 7 7 2 1 SAME 2>&1 |tee $LOG_DIR/ansor_depthwise_128_84_83_83_7_7_2_1_SAME.log
python3.7 -u $CODE_DIR/depthwise_tuning.py 128 672 11 11 3 3 1 1 SAME 2>&1 |tee $LOG_DIR/ansor_depthwise_128_672_11_11_3_3_1_1_SAME.log
python3.7 -u $CODE_DIR/depthwise_tuning.py 128 168 42 42 5 5 1 1 SAME 2>&1 |tee $LOG_DIR/ansor_depthwise_128_168_42_42_5_5_1_1_SAME.log
python3.7 -u $CODE_DIR/depthwise_tuning.py 128 672 21 21 5 5 2 1 SAME 2>&1 |tee $LOG_DIR/ansor_depthwise_128_672_21_21_5_5_2_1_SAME.log
python3.7 -u $CODE_DIR/depthwise_tuning.py 128 336 21 21 3 3 1 1 SAME 2>&1 |tee $LOG_DIR/ansor_depthwise_128_336_21_21_3_3_1_1_SAME.log
python3.7 -u $CODE_DIR/depthwise_tuning.py 128 672 21 21 7 7 2 1 SAME 2>&1 |tee $LOG_DIR/ansor_depthwise_128_672_21_21_7_7_2_1_SAME.log
python3.7 -u $CODE_DIR/depthwise_tuning.py 128 42 83 83 7 7 1 1 SAME 2>&1 |tee $LOG_DIR/ansor_depthwise_128_42_83_83_7_7_1_1_SAME.log
python3.7 -u $CODE_DIR/depthwise_tuning.py 128 84 42 42 7 7 1 1 SAME 2>&1 |tee $LOG_DIR/ansor_depthwise_128_84_42_42_7_7_1_1_SAME.log
python3.7 -u $CODE_DIR/depthwise_tuning.py 128 84 42 42 5 5 1 1 SAME 2>&1 |tee $LOG_DIR/ansor_depthwise_128_84_42_42_5_5_1_1_SAME.log
python3.7 -u $CODE_DIR/depthwise_tuning.py 128 168 42 42 3 3 1 1 SAME 2>&1 |tee $LOG_DIR/ansor_depthwise_128_168_42_42_3_3_1_1_SAME.log
python3.7 -u $CODE_DIR/depthwise_tuning.py 128 672 11 11 7 7 1 1 SAME 2>&1 |tee $LOG_DIR/ansor_depthwise_128_672_11_11_7_7_1_1_SAME.log
python3.7 -u $CODE_DIR/depthwise_tuning.py 128 336 42 42 5 5 2 1 SAME 2>&1 |tee $LOG_DIR/ansor_depthwise_128_336_42_42_5_5_2_1_SAME.log
python3.7 -u $CODE_DIR/depthwise_tuning.py 128 96 165 165 5 5 2 1 SAME 2>&1 |tee $LOG_DIR/ansor_depthwise_128_96_165_165_5_5_2_1_SAME.log
python3.7 -u $CODE_DIR/depthwise_tuning.py 128 336 21 21 7 7 1 1 SAME 2>&1 |tee $LOG_DIR/ansor_depthwise_128_336_21_21_7_7_1_1_SAME.log
python3.7 -u $CODE_DIR/depthwise_tuning.py 128 336 42 42 7 7 2 1 SAME 2>&1 |tee $LOG_DIR/ansor_depthwise_128_336_42_42_7_7_2_1_SAME.log
python3.7 -u $CODE_DIR/depthwise_tuning.py 128 42 83 83 3 3 1 1 SAME 2>&1 |tee $LOG_DIR/ansor_depthwise_128_42_83_83_3_3_1_1_SAME.log
python3.7 -u $CODE_DIR/depthwise_tuning.py 128 96 165 165 7 7 2 1 SAME 2>&1 |tee $LOG_DIR/ansor_depthwise_128_96_165_165_7_7_2_1_SAME.log
python3.7 -u $CODE_DIR/depthwise_tuning.py 128 84 42 42 3 3 1 1 SAME 2>&1 |tee $LOG_DIR/ansor_depthwise_128_84_42_42_3_3_1_1_SAME.log
python3.7 -u $CODE_DIR/depthwise_tuning.py 128 672 11 11 5 5 1 1 SAME 2>&1 |tee $LOG_DIR/ansor_depthwise_128_672_11_11_5_5_1_1_SAME.log
