# training bs 128
mkdir -p ./logs_v2_rocm/roller/wo_storage_align/pooling
LOG_DIR=./logs_v2_rocm/roller/wo_storage_align/pooling
CODE_DIR=.

python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir pooling --smem_tiling --reg_tiling --op avgpool2d_expr_S2P0 --shape 21504 42 42 1 1 2>&1 |tee $LOG_DIR/pooling0_avg_128_168_83_83_1_2_VALID.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir pooling --smem_tiling --reg_tiling --op avgpool2d_expr_S2P1 --shape 86016 11 11 3 3 2>&1 |tee $LOG_DIR/pooling1_avg_128_672_21_21_3_2_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir pooling --smem_tiling --reg_tiling --op avgpool2d_expr_S1P1 --shape 5376 83 83 3 3 2>&1 |tee $LOG_DIR/pooling2_avg_128_42_83_83_3_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir pooling --smem_tiling --reg_tiling --op avgpool2d_expr_S2P0 --shape 129024 21 21 1 1 2>&1 |tee $LOG_DIR/pooling3_avg_128_1008_42_42_1_2_VALID.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir pooling --smem_tiling --reg_tiling --op avgpool2d_expr_S2P1 --shape 43008 21 21 3 3 2>&1 |tee $LOG_DIR/pooling4_avg_128_336_42_42_3_2_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir pooling --smem_tiling --reg_tiling --op avgpool2d_expr_S2P1 --shape 10752 42 42 3 3 2>&1 |tee $LOG_DIR/pooling5_avg_128_84_83_83_3_2_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir pooling --smem_tiling --reg_tiling --op avgpool2d_expr_S1P1 --shape 86016 11 11 3 3 2>&1 |tee $LOG_DIR/pooling6_avg_128_672_11_11_3_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir pooling --smem_tiling --reg_tiling --op avgpool2d_expr_S2P0 --shape 12288 83 83 1 1 2>&1 |tee $LOG_DIR/pooling7_avg_128_96_165_165_1_2_VALID.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir pooling --smem_tiling --reg_tiling --op avgpool2d_expr_S2P0 --shape 258048 11 11 1 1 2>&1 |tee $LOG_DIR/pooling8_avg_128_2016_21_21_1_2_VALID.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir pooling --smem_tiling --reg_tiling --op avgpool2d_expr_S2P1 --shape 5376 83 83 3 3 2>&1 |tee $LOG_DIR/pooling9_avg_128_42_165_165_3_2_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir pooling --smem_tiling --reg_tiling --op avgpool2d_expr_S1P1 --shape 10752 42 42 3 3 2>&1 |tee $LOG_DIR/pooling10_avg_128_84_42_42_3_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir pooling --smem_tiling --reg_tiling --op avgpool2d_expr_S1P1 --shape 43008 21 21 3 3 2>&1 |tee $LOG_DIR/pooling11_avg_128_336_21_21_3_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir pooling --smem_tiling --reg_tiling --op avgpool2d_expr_S1P1 --shape 21504 42 42 3 3 2>&1 |tee $LOG_DIR/pooling12_avg_128_168_42_42_3_1_SAME.log

# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir pooling --smem_tiling --reg_tiling --op maxpool2d_expr_S2P1 --shape 86016 11 11 3 3 2>&1 |tee $LOG_DIR/pooling_max_128_672_21_21_3_2_SAME.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir pooling --smem_tiling --reg_tiling --op maxpool2d_expr_S2P1 --shape 43008 21 21 3 3 2>&1 |tee $LOG_DIR/pooling_max_128_336_42_42_3_2_SAME.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir pooling --smem_tiling --reg_tiling --op maxpool2d_expr_S2P1 --shape 8192 56 56 3 3 2>&1 |tee $LOG_DIR/pooling_max_128_64_112_112_3_2_SAME.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir pooling --smem_tiling --reg_tiling --op maxpool2d_expr_S2P1 --shape 10752 42 42 3 3 2>&1 |tee $LOG_DIR/pooling_max_128_84_83_83_3_2_SAME.log
# python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir pooling --smem_tiling --reg_tiling --op maxpool2d_expr_S2P1 --shape 5376 83 83 3 3 2>&1 |tee $LOG_DIR/pooling_max_128_42_165_165_3_2_SAME.log

