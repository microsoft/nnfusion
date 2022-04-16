# training bs 128
mkdir -p ./logs_v2_rocm/roller/wo_storage_align/elementwise
LOG_DIR=./logs_v2_rocm/roller/wo_storage_align/elementwise
CODE_DIR=.

python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 227598336 2>&1 |tee $LOG_DIR/relu0_128_1008_42_42.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 6422528 2>&1 |tee $LOG_DIR/relu1_128_256_14_14.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 25690112 2>&1 |tee $LOG_DIR/relu2_128_1024_14_14.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 12845056 2>&1 |tee $LOG_DIR/relu3_128_512_14_14.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 334540800 2>&1 |tee $LOG_DIR/relu4_128_96_165_165.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 75866112 2>&1 |tee $LOG_DIR/relu5_128_1344_21_21.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 41631744 2>&1 |tee $LOG_DIR/relu6_128_2688_11_11.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 102760448 2>&1 |tee $LOG_DIR/relu7_128_64_112_112.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 102760448 2>&1 |tee $LOG_DIR/relu8_128_256_56_56.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 12845056 2>&1 |tee $LOG_DIR/relu9_128_128_28_28.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 51380224 2>&1 |tee $LOG_DIR/relu10_128_512_28_28.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 25690112 2>&1 |tee $LOG_DIR/relu11_128_256_28_28.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 113799168 2>&1 |tee $LOG_DIR/relu12_128_2016_21_21.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 10407936 2>&1 |tee $LOG_DIR/relu13_128_672_11_11.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 148141056 2>&1 |tee $LOG_DIR/relu14_128_168_83_83.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 25690112 2>&1 |tee $LOG_DIR/relu15_128_64_56_56.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 37933056 2>&1 |tee $LOG_DIR/relu16_128_168_42_42.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 18966528 2>&1 |tee $LOG_DIR/relu17_128_336_21_21.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 62447616 2>&1 |tee $LOG_DIR/relu18_128_4032_11_11.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 3211264 2>&1 |tee $LOG_DIR/relu19_128_512_7_7.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 12845056 2>&1 |tee $LOG_DIR/relu20_128_2048_7_7.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 74070528 2>&1 |tee $LOG_DIR/relu21_128_84_83_83.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 75866112 2>&1 |tee $LOG_DIR/relu22_128_336_42_42.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 146361600 2>&1 |tee $LOG_DIR/relu23_128_42_165_165.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 37933056 2>&1 |tee $LOG_DIR/relu24_128_672_21_21.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 18966528 2>&1 |tee $LOG_DIR/relu25_128_84_42_42.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 37035264 2>&1 |tee $LOG_DIR/relu26_128_42_83_83.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 51380224 2>&1 |tee $LOG_DIR/relu27_128_128_56_56.log

# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 128 1008 42 42 2>&1 |tee $LOG_DIR/relu_128_1008_42_42.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 128 256 14 14 2>&1 |tee $LOG_DIR/relu_128_256_14_14.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 128 1024 14 14 2>&1 |tee $LOG_DIR/relu_128_1024_14_14.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 128 512 14 14 2>&1 |tee $LOG_DIR/relu_128_512_14_14.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 128 96 165 165 2>&1 |tee $LOG_DIR/relu_128_96_165_165.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 128 1344 21 21 2>&1 |tee $LOG_DIR/relu_128_1344_21_21.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 128 2688 11 11 2>&1 |tee $LOG_DIR/relu_128_2688_11_11.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 128 64 112 112 2>&1 |tee $LOG_DIR/relu_128_64_112_112.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 128 256 56 56 2>&1 |tee $LOG_DIR/relu_128_256_56_56.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 128 128 28 28 2>&1 |tee $LOG_DIR/relu_128_128_28_28.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 128 512 28 28 2>&1 |tee $LOG_DIR/relu_128_512_28_28.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 128 256 28 28 2>&1 |tee $LOG_DIR/relu_128_256_28_28.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 128 2016 21 21 2>&1 |tee $LOG_DIR/relu_128_2016_21_21.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 128 672 11 11 2>&1 |tee $LOG_DIR/relu_128_672_11_11.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 128 168 83 83 2>&1 |tee $LOG_DIR/relu_128_168_83_83.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 128 64 56 56 2>&1 |tee $LOG_DIR/relu_128_64_56_56.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 128 168 42 42 2>&1 |tee $LOG_DIR/relu_128_168_42_42.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 128 336 21 21 2>&1 |tee $LOG_DIR/relu_128_336_21_21.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 128 4032 11 11 2>&1 |tee $LOG_DIR/relu_128_4032_11_11.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 128 512 7 7 2>&1 |tee $LOG_DIR/relu_128_512_7_7.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 128 2048 7 7 2>&1 |tee $LOG_DIR/relu_128_2048_7_7.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 128 84 83 83 2>&1 |tee $LOG_DIR/relu_128_84_83_83.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 128 336 42 42 2>&1 |tee $LOG_DIR/relu_128_336_42_42.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 128 42 165 165 2>&1 |tee $LOG_DIR/relu_128_42_165_165.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 128 672 21 21 2>&1 |tee $LOG_DIR/relu_128_672_21_21.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 128 84 42 42 2>&1 |tee $LOG_DIR/relu_128_84_42_42.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 128 42 83 83 2>&1 |tee $LOG_DIR/relu_128_42_83_83.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir elementwise --op relu_expr --shape 128 128 56 56 2>&1 |tee $LOG_DIR/relu_128_128_56_56.log