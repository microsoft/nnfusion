mkdir -p ./logs/elementwise
CODE_DIR=./src
LOG_DIR=./logs/elementwise
REPEAT_TIME=10000
python3 -u $CODE_DIR/element.py 128 1008 42 42 $REPEAT_TIME 2>&1 |tee $LOG_DIR/elementwise0_128_1008_42_42.log
python3 -u $CODE_DIR/element.py 128 256 14 14 $REPEAT_TIME 2>&1 |tee $LOG_DIR/elementwise1_128_256_14_14.log
python3 -u $CODE_DIR/element.py 128 1024 14 14 $REPEAT_TIME 2>&1 |tee $LOG_DIR/elementwise2_128_1024_14_14.log
python3 -u $CODE_DIR/element.py 128 512 14 14 $REPEAT_TIME 2>&1 |tee $LOG_DIR/elementwise3_128_512_14_14.log
python3 -u $CODE_DIR/element.py 128 96 165 165 $REPEAT_TIME 2>&1 |tee $LOG_DIR/elementwise4_128_96_165_165.log
python3 -u $CODE_DIR/element.py 128 1344 21 21 $REPEAT_TIME 2>&1 |tee $LOG_DIR/elementwise5_128_1344_21_21.log
python3 -u $CODE_DIR/element.py 128 2688 11 11 $REPEAT_TIME 2>&1 |tee $LOG_DIR/elementwise6_128_2688_11_11.log
python3 -u $CODE_DIR/element.py 128 64 112 112 $REPEAT_TIME 2>&1 |tee $LOG_DIR/elementwise7_128_64_112_112.log
python3 -u $CODE_DIR/element.py 128 256 56 56 $REPEAT_TIME 2>&1 |tee $LOG_DIR/elementwise8_128_256_56_56.log
python3 -u $CODE_DIR/element.py 128 128 28 28 $REPEAT_TIME 2>&1 |tee $LOG_DIR/elementwise9_128_128_28_28.log
python3 -u $CODE_DIR/element.py 128 512 28 28 $REPEAT_TIME 2>&1 |tee $LOG_DIR/elementwise10_128_512_28_28.log
python3 -u $CODE_DIR/element.py 128 256 28 28 $REPEAT_TIME 2>&1 |tee $LOG_DIR/elementwise11_128_256_28_28.log
python3 -u $CODE_DIR/element.py 128 2016 21 21 $REPEAT_TIME 2>&1 |tee $LOG_DIR/elementwise12_128_2016_21_21.log
python3 -u $CODE_DIR/element.py 128 672 11 11 $REPEAT_TIME 2>&1 |tee $LOG_DIR/elementwise13_128_672_11_11.log
python3 -u $CODE_DIR/element.py 128 168 83 83 $REPEAT_TIME 2>&1 |tee $LOG_DIR/elementwise14_128_168_83_83.log
python3 -u $CODE_DIR/element.py 128 64 56 56 $REPEAT_TIME 2>&1 |tee $LOG_DIR/elementwise15_128_64_56_56.log
python3 -u $CODE_DIR/element.py 128 168 42 42 $REPEAT_TIME 2>&1 |tee $LOG_DIR/elementwise16_128_168_42_42.log
python3 -u $CODE_DIR/element.py 128 336 21 21 $REPEAT_TIME 2>&1 |tee $LOG_DIR/elementwise17_128_336_21_21.log
python3 -u $CODE_DIR/element.py 128 4032 11 11 $REPEAT_TIME 2>&1 |tee $LOG_DIR/elementwise18_128_4032_11_11.log
python3 -u $CODE_DIR/element.py 128 512 7 7 $REPEAT_TIME 2>&1 |tee $LOG_DIR/elementwise19_128_512_7_7.log
python3 -u $CODE_DIR/element.py 128 2048 7 7 $REPEAT_TIME 2>&1 |tee $LOG_DIR/elementwise20_128_2048_7_7.log
python3 -u $CODE_DIR/element.py 128 84 83 83 $REPEAT_TIME 2>&1 |tee $LOG_DIR/elementwise21_128_84_83_83.log
python3 -u $CODE_DIR/element.py 128 336 42 42 $REPEAT_TIME 2>&1 |tee $LOG_DIR/elementwise22_128_336_42_42.log
python3 -u $CODE_DIR/element.py 128 42 165 165 $REPEAT_TIME 2>&1 |tee $LOG_DIR/elementwise23_128_42_165_165.log
python3 -u $CODE_DIR/element.py 128 672 21 21 $REPEAT_TIME 2>&1 |tee $LOG_DIR/elementwise24_128_672_21_21.log
python3 -u $CODE_DIR/element.py 128 84 42 42 $REPEAT_TIME 2>&1 |tee $LOG_DIR/elementwise25_128_84_42_42.log
python3 -u $CODE_DIR/element.py 128 42 83 83 $REPEAT_TIME 2>&1 |tee $LOG_DIR/elementwise26_128_42_83_83.log
python3 -u $CODE_DIR/element.py 128 128 56 56 $REPEAT_TIME 2>&1 |tee $LOG_DIR/elementwise27_128_128_56_56.log
