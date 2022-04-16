# training bs 128
LOG_DIR=logs/ansor/pooling
CODE_DIR=microbenchmark/tvm/ansor

python3.7 -u $CODE_DIR/pooling_tuning.py avg 128 168 83 83 1 2 VALID 2>&1 |tee $LOG_DIR/ansor_pooling_avg_128_168_83_83_1_2_VALID.log
python3.7 -u $CODE_DIR/pooling_tuning.py avg 128 672 21 21 3 2 SAME 2>&1 |tee $LOG_DIR/ansor_pooling_avg_128_672_21_21_3_2_SAME.log
python3.7 -u $CODE_DIR/pooling_tuning.py avg 128 42 83 83 3 1 SAME 2>&1 |tee $LOG_DIR/ansor_pooling_avg_128_42_83_83_3_1_SAME.log
python3.7 -u $CODE_DIR/pooling_tuning.py avg 128 1008 42 42 1 2 VALID 2>&1 |tee $LOG_DIR/ansor_pooling_avg_128_1008_42_42_1_2_VALID.log
python3.7 -u $CODE_DIR/pooling_tuning.py avg 128 336 42 42 3 2 SAME 2>&1 |tee $LOG_DIR/ansor_pooling_avg_128_336_42_42_3_2_SAME.log
python3.7 -u $CODE_DIR/pooling_tuning.py avg 128 84 83 83 3 2 SAME 2>&1 |tee $LOG_DIR/ansor_pooling_avg_128_84_83_83_3_2_SAME.log
python3.7 -u $CODE_DIR/pooling_tuning.py avg 128 672 11 11 3 1 SAME 2>&1 |tee $LOG_DIR/ansor_pooling_avg_128_672_11_11_3_1_SAME.log
python3.7 -u $CODE_DIR/pooling_tuning.py avg 128 96 165 165 1 2 VALID 2>&1 |tee $LOG_DIR/ansor_pooling_avg_128_96_165_165_1_2_VALID.log
python3.7 -u $CODE_DIR/pooling_tuning.py avg 128 2016 21 21 1 2 VALID 2>&1 |tee $LOG_DIR/ansor_pooling_avg_128_2016_21_21_1_2_VALID.log
python3.7 -u $CODE_DIR/pooling_tuning.py avg 128 42 165 165 3 2 SAME 2>&1 |tee $LOG_DIR/ansor_pooling_avg_128_42_165_165_3_2_SAME.log
python3.7 -u $CODE_DIR/pooling_tuning.py avg 128 84 42 42 3 1 SAME 2>&1 |tee $LOG_DIR/ansor_pooling_avg_84_42_42_3_1_SAME.log
python3.7 -u $CODE_DIR/pooling_tuning.py avg 128 336 21 21 3 1 SAME 2>&1 |tee $LOG_DIR/ansor_pooling_avg_128_336_21_21_3_1_SAME.log
python3.7 -u $CODE_DIR/pooling_tuning.py avg 128 168 42 42 3 1 SAME 2>&1 |tee $LOG_DIR/ansor_pooling_avg_128_168_42_42_3_1_SAME.log

python3.7 -u $CODE_DIR/pooling_tuning.py max 128 672 21 21 3 2 SAME 2>&1 |tee $LOG_DIR/ansor_pooling_max_128_672_21_21_3_2_SAME.log
python3.7 -u $CODE_DIR/pooling_tuning.py max 128 336 42 42 3 2 SAME 2>&1 |tee $LOG_DIR/ansor_pooling_max_128_336_42_42_3_2_SAME.log
python3.7 -u $CODE_DIR/pooling_tuning.py max 128 64 112 112 3 2 SAME 2>&1 |tee $LOG_DIR/ansor_pooling_max_128_64_112_112_3_2_SAME.log
python3.7 -u $CODE_DIR/pooling_tuning.py max 128 84 83 83 3 2 SAME 2>&1 |tee $LOG_DIR/ansor_pooling_max_128_84_83_83_3_2_SAME.log
python3.7 -u $CODE_DIR/pooling_tuning.py max 128 42 165 165 3 2 SAME 2>&1 |tee $LOG_DIR/ansor_pooling_max_128_42_165_165_3_2_SAME.log
