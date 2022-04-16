# training bs 128
LOG_DIR=logs/compile-time
CODE_DIR=compile-time

python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 128 28 28 128 3 3 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_128_28_28_128_3_3_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 128 58 58 128 3 3 2 1 VALID > $LOG_DIR/implicit_gemm_padding_roller_128_128_58_58_128_3_3_2_1_VALID.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 256 30 30 256 3 3 2 1 VALID > $LOG_DIR/implicit_gemm_padding_roller_128_256_30_30_256_3_3_2_1_VALID.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 168 42 42 168 1 1 1 1 VALID > $LOG_DIR/implicit_gemm_padding_roller_128_168_42_42_168_1_1_1_1_VALID.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 512 7 7 512 3 3 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_512_7_7_512_3_3_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 256 14 14 1024 1 1 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_256_14_14_1024_1_1_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 1024 14 14 256 1 1 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_1024_14_14_256_1_1_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 1024 14 14 512 1 1 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_1024_14_14_512_1_1_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 1008 21 21 168 1 1 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_1008_21_21_168_1_1_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 42 83 83 42 1 1 1 1 VALID > $LOG_DIR/implicit_gemm_padding_roller_128_42_83_83_42_1_1_1_1_VALID.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 4032 11 11 672 1 1 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_4032_11_11_672_1_1_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 512 16 16 512 3 3 2 1 VALID > $LOG_DIR/implicit_gemm_padding_roller_128_512_16_16_512_3_3_2_1_VALID.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 96 83 83 42 1 1 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_96_83_83_42_1_1_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 96 165 165 42 1 1 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_96_165_165_42_1_1_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 168 83 83 84 1 1 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_168_83_83_84_1_1_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 336 21 21 336 1 1 1 1 VALID > $LOG_DIR/implicit_gemm_padding_roller_128_336_21_21_336_1_1_1_1_VALID.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 512 28 28 1024 1 1 2 1 VALID > $LOG_DIR/implicit_gemm_padding_roller_128_512_28_28_1024_1_1_2_1_VALID.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 64 56 56 256 1 1 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_64_56_56_256_1_1_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 256 56 56 64 1 1 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_256_56_56_64_1_1_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 128 28 28 512 1 1 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_128_28_28_512_1_1_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 512 28 28 128 1 1 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_512_28_28_128_1_1_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 168 42 42 84 1 1 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_168_42_42_84_1_1_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 512 28 28 256 1 1 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_512_28_28_256_1_1_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 64 56 56 64 3 3 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_64_56_56_64_3_3_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 2016 21 21 672 1 1 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_2016_21_21_672_1_1_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 512 7 7 2048 1 1 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_512_7_7_2048_1_1_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 2048 7 7 512 1 1 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_2048_7_7_512_1_1_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 84 42 42 84 1 1 1 1 VALID > $LOG_DIR/implicit_gemm_padding_roller_128_84_42_42_84_1_1_1_1_VALID.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 336 42 42 168 1 1 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_336_42_42_168_1_1_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 672 11 11 672 1 1 1 1 VALID > $LOG_DIR/implicit_gemm_padding_roller_128_672_11_11_672_1_1_1_1_VALID.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 1024 14 14 2048 1 1 2 1 VALID > $LOG_DIR/implicit_gemm_padding_roller_128_1024_14_14_2048_1_1_2_1_VALID.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 2016 11 11 336 1 1 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_2016_11_11_336_1_1_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 2016 21 21 336 1 1 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_2016_21_21_336_1_1_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 1008 42 42 336 1 1 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_1008_42_42_336_1_1_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 64 56 56 64 1 1 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_64_56_56_64_1_1_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 3 230 230 64 7 7 2 1 VALID > $LOG_DIR/implicit_gemm_padding_roller_128_3_230_230_64_7_7_2_1_VALID.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 3 331 331 96 3 3 2 1 VALID > $LOG_DIR/implicit_gemm_padding_roller_128_3_331_331_96_3_3_2_1_VALID.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 256 56 56 128 1 1 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_256_56_56_128_1_1_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 256 14 14 256 3 3 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_256_14_14_256_3_3_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 2688 11 11 672 1 1 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_2688_11_11_672_1_1_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 1008 42 42 168 1 1 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_1008_42_42_168_1_1_1_1_SAME.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 96 83 83 42 1 1 1 1 VALID > $LOG_DIR/implicit_gemm_padding_roller_128_96_83_83_42_1_1_1_1_VALID.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 256 56 56 512 1 1 2 1 VALID > $LOG_DIR/implicit_gemm_padding_roller_128_256_56_56_512_1_1_2_1_VALID.log
python3 $CODE_DIR/test_implicit_gemm_padding_roller_time.py 128 1344 21 21 336 1 1 1 1 SAME > $LOG_DIR/implicit_gemm_padding_roller_128_1344_21_21_336_1_1_1_1_SAME.log
