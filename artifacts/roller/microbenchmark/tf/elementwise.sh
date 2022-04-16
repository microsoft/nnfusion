mkdir -p ./logs/tf/elementwise
CODE_DIR=./microbenchmark/tf/src
LOG_DIR=./logs/tf/elementwise
REAPEAT_TIME=10000
python3 -u $CODE_DIR/element.py 128 1008 42 42 $REPEAT_TIME 2>&1 |tee $LOG_DIR/relu0_128_1008_42_42.log
python3 -u $CODE_DIR/element.py 128 256 14 14 $REPEAT_TIME 2>&1 |tee $LOG_DIR/relu1_128_256_14_14.log
python3 -u $CODE_DIR/element.py 128 1024 14 14 $REPEAT_TIME 2>&1 |tee $LOG_DIR/relu2_128_1024_14_14.log
python3 -u $CODE_DIR/element.py 128 512 14 14 $REPEAT_TIME 2>&1 |tee $LOG_DIR/relu3_128_512_14_14.log
python3 -u $CODE_DIR/element.py 128 96 165 165 $REPEAT_TIME 2>&1 |tee $LOG_DIR/relu4_128_96_165_165.log
python3 -u $CODE_DIR/element.py 128 1344 21 21 $REPEAT_TIME 2>&1 |tee $LOG_DIR/relu5_128_1344_21_21.log
python3 -u $CODE_DIR/element.py 128 2688 11 11 $REPEAT_TIME 2>&1 |tee $LOG_DIR/relu6_128_2688_11_11.log
python3 -u $CODE_DIR/element.py 128 64 112 112 $REPEAT_TIME 2>&1 |tee $LOG_DIR/relu7_128_64_112_112.log
python3 -u $CODE_DIR/element.py 128 256 56 56 $REPEAT_TIME 2>&1 |tee $LOG_DIR/relu8_128_256_56_56.log
python3 -u $CODE_DIR/element.py 128 128 28 28 $REPEAT_TIME 2>&1 |tee $LOG_DIR/relu9_128_128_28_28.log
python3 -u $CODE_DIR/element.py 128 512 28 28 $REPEAT_TIME 2>&1 |tee $LOG_DIR/relu10_128_512_28_28.log
python3 -u $CODE_DIR/element.py 128 256 28 28 $REPEAT_TIME 2>&1 |tee $LOG_DIR/relu11_128_256_28_28.log
python3 -u $CODE_DIR/element.py 128 2016 21 21 $REPEAT_TIME 2>&1 |tee $LOG_DIR/relu12_128_2016_21_21.log
python3 -u $CODE_DIR/element.py 128 672 11 11 $REPEAT_TIME 2>&1 |tee $LOG_DIR/relu13_128_672_11_11.log
python3 -u $CODE_DIR/element.py 128 168 83 83 $REPEAT_TIME 2>&1 |tee $LOG_DIR/relu14_128_168_83_83.log
python3 -u $CODE_DIR/element.py 128 64 56 56 $REPEAT_TIME 2>&1 |tee $LOG_DIR/relu15_128_64_56_56.log
python3 -u $CODE_DIR/element.py 128 168 42 42 $REPEAT_TIME 2>&1 |tee $LOG_DIR/relu16_128_168_42_42.log
python3 -u $CODE_DIR/element.py 128 336 21 21 $REPEAT_TIME 2>&1 |tee $LOG_DIR/relu17_128_336_21_21.log
python3 -u $CODE_DIR/element.py 128 4032 11 11 $REPEAT_TIME 2>&1 |tee $LOG_DIR/relu18_128_4032_11_11.log
python3 -u $CODE_DIR/element.py 128 512 7 7 $REPEAT_TIME 2>&1 |tee $LOG_DIR/relu19_128_512_7_7.log
python3 -u $CODE_DIR/element.py 128 2048 7 7 $REPEAT_TIME 2>&1 |tee $LOG_DIR/relu20_128_2048_7_7.log
python3 -u $CODE_DIR/element.py 128 84 83 83 $REPEAT_TIME 2>&1 |tee $LOG_DIR/relu21_128_84_83_83.log
python3 -u $CODE_DIR/element.py 128 336 42 42 $REPEAT_TIME 2>&1 |tee $LOG_DIR/relu22_128_336_42_42.log
python3 -u $CODE_DIR/element.py 128 42 165 165 $REPEAT_TIME 2>&1 |tee $LOG_DIR/relu23_128_42_165_165.log
python3 -u $CODE_DIR/element.py 128 672 21 21 $REPEAT_TIME 2>&1 |tee $LOG_DIR/relu24_128_672_21_21.log
python3 -u $CODE_DIR/element.py 128 84 42 42 $REPEAT_TIME 2>&1 |tee $LOG_DIR/relu25_128_84_42_42.log
python3 -u $CODE_DIR/element.py 128 42 83 83 $REPEAT_TIME 2>&1 |tee $LOG_DIR/relu26_128_42_83_83.log
python3 -u $CODE_DIR/element.py 128 128 56 56 $REPEAT_TIME 2>&1 |tee $LOG_DIR/relu27_128_128_56_56.log
