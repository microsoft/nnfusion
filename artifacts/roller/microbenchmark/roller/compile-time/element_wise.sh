# training bs 128
LOG_DIR=logs/compile-time
CODE_DIR=compile-time

python3 $CODE_DIR/test_elementwise_roller.py relu 128 1008 42 42 > $LOG_DIR/relu_128_1008_42_42.log
python3 $CODE_DIR/test_elementwise_roller.py relu 128 256 14 14 > $LOG_DIR/relu_128_256_14_14.log
python3 $CODE_DIR/test_elementwise_roller.py relu 128 1024 14 14 > $LOG_DIR/relu_128_1024_14_14.log
python3 $CODE_DIR/test_elementwise_roller.py relu 128 512 14 14 > $LOG_DIR/relu_128_512_14_14.log
python3 $CODE_DIR/test_elementwise_roller.py relu 128 96 165 165 > $LOG_DIR/relu_128_96_165_165.log
python3 $CODE_DIR/test_elementwise_roller.py relu 128 1344 21 21 > $LOG_DIR/relu_128_1344_21_21.log
python3 $CODE_DIR/test_elementwise_roller.py relu 128 2688 11 11 > $LOG_DIR/relu_128_2688_11_11.log
python3 $CODE_DIR/test_elementwise_roller.py relu 128 64 112 112 > $LOG_DIR/relu_128_64_112_112.log
python3 $CODE_DIR/test_elementwise_roller.py relu 128 256 56 56 > $LOG_DIR/relu_128_256_56_56.log
python3 $CODE_DIR/test_elementwise_roller.py relu 128 128 28 28 > $LOG_DIR/relu_128_128_28_28.log
python3 $CODE_DIR/test_elementwise_roller.py relu 128 512 28 28 > $LOG_DIR/relu_128_512_28_28.log
python3 $CODE_DIR/test_elementwise_roller.py relu 128 256 28 28 > $LOG_DIR/relu_128_256_28_28.log
python3 $CODE_DIR/test_elementwise_roller.py relu 128 2016 21 21 > $LOG_DIR/relu_128_2016_21_21.log
python3 $CODE_DIR/test_elementwise_roller.py relu 128 672 11 11 > $LOG_DIR/relu_128_672_11_11.log
python3 $CODE_DIR/test_elementwise_roller.py relu 128 168 83 83 > $LOG_DIR/relu_128_168_83_83.log
python3 $CODE_DIR/test_elementwise_roller.py relu 128 64 56 56 > $LOG_DIR/relu_128_64_56_56.log
python3 $CODE_DIR/test_elementwise_roller.py relu 128 168 42 42 > $LOG_DIR/relu_128_168_42_42.log
python3 $CODE_DIR/test_elementwise_roller.py relu 128 336 21 21 > $LOG_DIR/relu_128_336_21_21.log
python3 $CODE_DIR/test_elementwise_roller.py relu 128 4032 11 11 > $LOG_DIR/relu_128_4032_11_11.log
python3 $CODE_DIR/test_elementwise_roller.py relu 128 512 7 7 > $LOG_DIR/relu_128_512_7_7.log
python3 $CODE_DIR/test_elementwise_roller.py relu 128 2048 7 7 > $LOG_DIR/relu_128_2048_7_7.log
python3 $CODE_DIR/test_elementwise_roller.py relu 128 84 83 83 > $LOG_DIR/relu_128_84_83_83.log
python3 $CODE_DIR/test_elementwise_roller.py relu 128 336 42 42 > $LOG_DIR/relu_128_336_42_42.log
python3 $CODE_DIR/test_elementwise_roller.py relu 128 42 165 165 > $LOG_DIR/relu_128_42_165_165.log
python3 $CODE_DIR/test_elementwise_roller.py relu 128 672 21 21 > $LOG_DIR/relu_128_672_21_21.log
python3 $CODE_DIR/test_elementwise_roller.py relu 128 84 42 42 > $LOG_DIR/relu_128_84_42_42.log
python3 $CODE_DIR/test_elementwise_roller.py relu 128 42 83 83 > $LOG_DIR/relu_128_42_83_83.log
python3 $CODE_DIR/test_elementwise_roller.py relu 128 128 56 56 > $LOG_DIR/relu_128_128_56_56.log
