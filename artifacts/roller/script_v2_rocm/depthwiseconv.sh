# training bs 128
mkdir -p ./logs_v2_rocm/roller/wo_storage_align/depthwiseconv
LOG_DIR=./logs_v2_rocm/roller/wo_storage_align/depthwiseconv
CODE_DIR=.

python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir depthwiseconv --schedule_fuse --smem_tiling --reg_tiling --op depthwiseconv_expr_S2D1P2 --shape 128 84 42 42 5 5 2>&1 |tee $LOG_DIR/depthwiseconv0_128_84_83_83_5_2_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir depthwiseconv --schedule_fuse --smem_tiling --reg_tiling --op depthwiseconv_expr_S1D1P2 --shape 128 42 83 83 5 5 2>&1 |tee $LOG_DIR/depthwiseconv1_128_42_83_83_5_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir depthwiseconv --schedule_fuse --smem_tiling --reg_tiling --op depthwiseconv_expr_S1D1P2 --shape 128 336 21 21 5 5 2>&1 |tee $LOG_DIR/depthwiseconv2_128_336_21_21_5_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir depthwiseconv --schedule_fuse --smem_tiling --reg_tiling --op depthwiseconv_expr_S2D1P2 --shape 128 42 83 83 5 5 2>&1 |tee $LOG_DIR/depthwiseconv3_128_42_165_165_5_2_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir depthwiseconv --schedule_fuse --smem_tiling --reg_tiling --op depthwiseconv_expr_S2D1P3 --shape 128 84 42 42 7 7 2>&1 |tee $LOG_DIR/depthwiseconv4_128_84_83_83_7_2_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir depthwiseconv --schedule_fuse --smem_tiling --reg_tiling --op depthwiseconv_expr_S1D1P1 --shape 128 672 11 11 3 3 2>&1 |tee $LOG_DIR/depthwiseconv5_128_672_11_11_3_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir depthwiseconv --schedule_fuse --smem_tiling --reg_tiling --op depthwiseconv_expr_S1D1P2 --shape 128 168 42 42 5 5 2>&1 |tee $LOG_DIR/depthwiseconv6_128_168_42_42_5_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir depthwiseconv --schedule_fuse --smem_tiling --reg_tiling --op depthwiseconv_expr_S2D1P2 --shape 128 672 11 11 5 5 2>&1 |tee $LOG_DIR/depthwiseconv7_128_672_21_21_5_2_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir depthwiseconv --schedule_fuse --smem_tiling --reg_tiling --op depthwiseconv_expr_S1D1P1 --shape 128 336 21 21 3 3 2>&1 |tee $LOG_DIR/depthwiseconv8_128_336_21_21_3_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir depthwiseconv --schedule_fuse --smem_tiling --reg_tiling --op depthwiseconv_expr_S2D1P3 --shape 128 672 11 11 7 7 2>&1 |tee $LOG_DIR/depthwiseconv9_128_672_21_21_7_2_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir depthwiseconv --schedule_fuse --smem_tiling --reg_tiling --op depthwiseconv_expr_S1D1P3 --shape 128 42 83 83 7 7 2>&1 |tee $LOG_DIR/depthwiseconv10_128_42_83_83_7_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir depthwiseconv --schedule_fuse --smem_tiling --reg_tiling --op depthwiseconv_expr_S1D1P3 --shape 128 84 42 42 7 7 2>&1 |tee $LOG_DIR/depthwiseconv11_128_84_42_42_7_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir depthwiseconv --schedule_fuse --smem_tiling --reg_tiling --op depthwiseconv_expr_S1D1P2 --shape 128 84 42 42 5 5 2>&1 |tee $LOG_DIR/depthwiseconv12_128_84_42_42_5_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir depthwiseconv --schedule_fuse --smem_tiling --reg_tiling --op depthwiseconv_expr_S1D1P1 --shape 128 168 42 42 3 3 2>&1 |tee $LOG_DIR/depthwiseconv13_128_168_42_42_3_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir depthwiseconv --schedule_fuse --smem_tiling --reg_tiling --op depthwiseconv_expr_S1D1P3 --shape 128 672 11 11 7 7 2>&1 |tee $LOG_DIR/depthwiseconv14_128_672_11_11_7_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir depthwiseconv --schedule_fuse --smem_tiling --reg_tiling --op depthwiseconv_expr_S2D1P2 --shape 128 336 21 21 5 5 2>&1 |tee $LOG_DIR/depthwiseconv15_128_336_42_42_5_2_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir depthwiseconv --schedule_fuse --smem_tiling --reg_tiling --op depthwiseconv_expr_S2D1P2 --shape 128 96 83 83 5 5 2>&1 |tee $LOG_DIR/depthwiseconv16_128_96_165_165_5_2_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir depthwiseconv --schedule_fuse --smem_tiling --reg_tiling --op depthwiseconv_expr_S1D1P3 --shape 128 336 21 21 7 7 2>&1 |tee $LOG_DIR/depthwiseconv17_128_336_21_21_7_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir depthwiseconv --schedule_fuse --smem_tiling --reg_tiling --op depthwiseconv_expr_S2D1P3 --shape 128 336 21 21 7 7 2>&1 |tee $LOG_DIR/depthwiseconv18_128_336_42_42_7_2_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir depthwiseconv --schedule_fuse --smem_tiling --reg_tiling --op depthwiseconv_expr_S1D1P1 --shape 128 42 83 83 3 3 2>&1 |tee $LOG_DIR/depthwiseconv19_128_42_83_83_3_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir depthwiseconv --schedule_fuse --smem_tiling --reg_tiling --op depthwiseconv_expr_S2D1P3 --shape 128 96 83 83 7 7 2>&1 |tee $LOG_DIR/depthwiseconv20_128_96_165_165_7_2_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir depthwiseconv --schedule_fuse --smem_tiling --reg_tiling --op depthwiseconv_expr_S1D1P1 --shape 128 84 42 42 3 3 2>&1 |tee $LOG_DIR/depthwiseconv21_128_84_42_42_3_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir depthwiseconv --schedule_fuse --smem_tiling --reg_tiling --op depthwiseconv_expr_S1D1P2 --shape 128 672 11 11 5 5 2>&1 |tee $LOG_DIR/depthwiseconv22_128_672_11_11_5_1_1_SAME.log
