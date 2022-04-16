# training bs 128
mkdir -p ./logs_v2_rocm/roller/wo_storage_align/conv
LOG_DIR=./logs_v2_rocm/roller/wo_storage_align/conv
CODE_DIR=.

python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P1 --shape 128 128 28 28 128 3 3 2>&1 |tee $LOG_DIR/conv0_128_128_28_28_128_3_3_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S2D1P0 --shape 128 128 28 28 128 3 3 2>&1 |tee $LOG_DIR/conv1_128_128_58_58_128_3_3_2_1_VALID.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S2D1P0 --shape 128 256 14 14 256 3 3 2>&1 |tee $LOG_DIR/conv2_128_256_30_30_256_3_3_2_1_VALID.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 168 42 42 168 1 1 2>&1 |tee $LOG_DIR/conv3_128_168_42_42_168_1_1_1_1_VALID.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P1 --shape 128 512 7 7 512 3 3 2>&1 |tee $LOG_DIR/conv4_128_512_7_7_512_3_3_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 1024 14 14 256 1 1 2>&1 |tee $LOG_DIR/conv5_128_256_14_14_1024_1_1_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 256 14 14 1024 1 1 2>&1 |tee $LOG_DIR/conv6_128_1024_14_14_256_1_1_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 512 14 14 1024 1 1 2>&1 |tee $LOG_DIR/conv7_128_1024_14_14_512_1_1_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 168 21 21 1008 1 1 2>&1 |tee $LOG_DIR/conv8_128_1008_21_21_168_1_1_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 42 83 83 42 1 1 2>&1 |tee $LOG_DIR/conv9_128_42_83_83_42_1_1_1_1_VALID.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 672 11 11 4032 1 1 2>&1 |tee $LOG_DIR/conv10_128_4032_11_11_672_1_1_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S2D1P0 --shape 128 512 7 7 512 3 3 2>&1 |tee $LOG_DIR/conv11_128_512_16_16_512_3_3_2_1_VALID.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 42 83 83 96 1 1 2>&1 |tee $LOG_DIR/conv12_128_96_83_83_42_1_1_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 42 165 165 96 1 1 2>&1 |tee $LOG_DIR/conv13_128_96_165_165_42_1_1_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 84 83 83 168 1 1 2>&1 |tee $LOG_DIR/conv14_128_168_83_83_84_1_1_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 336 21 21 336 1 1 2>&1 |tee $LOG_DIR/conv15_128_336_21_21_336_1_1_1_1_VALID.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S2D1P0 --shape 128 1024 14 14 512 1 1 2>&1 |tee $LOG_DIR/conv16_128_512_28_28_1024_1_1_2_1_VALID.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 256 56 56 64 1 1 2>&1 |tee $LOG_DIR/conv17_128_64_56_56_256_1_1_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 64 56 56 256 1 1 2>&1 |tee $LOG_DIR/conv18_128_256_56_56_64_1_1_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 512 28 28 128 1 1 2>&1 |tee $LOG_DIR/conv19_128_128_28_28_512_1_1_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 128 28 28 512 1 1 2>&1 |tee $LOG_DIR/conv20_128_512_28_28_128_1_1_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 84 42 42 168 1 1 2>&1 |tee $LOG_DIR/conv21_128_168_42_42_84_1_1_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 256 28 28 512 1 1 2>&1 |tee $LOG_DIR/conv22_128_512_28_28_256_1_1_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P1 --shape 128 64 56 56 64 3 3 2>&1 |tee $LOG_DIR/conv23_128_64_56_56_64_3_3_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 672 21 21 2016 1 1 2>&1 |tee $LOG_DIR/conv24_128_2016_21_21_672_1_1_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 2048 7 7 512 1 1 2>&1 |tee $LOG_DIR/conv25_128_512_7_7_2048_1_1_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 512 7 7 2048 1 1 2>&1 |tee $LOG_DIR/conv26_128_2048_7_7_512_1_1_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 84 42 42 84 1 1 2>&1 |tee $LOG_DIR/conv27_128_84_42_42_84_1_1_1_1_VALID.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 168 42 42 336 1 1 2>&1 |tee $LOG_DIR/conv28_128_336_42_42_168_1_1_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 672 11 11 672 1 1 2>&1 |tee $LOG_DIR/conv29_128_672_11_11_672_1_1_1_1_VALID.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S2D1P0 --shape 128 2048 7 7 1024 1 1 2>&1 |tee $LOG_DIR/conv30_128_1024_14_14_2048_1_1_2_1_VALID.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 336 11 11 2016 1 1 2>&1 |tee $LOG_DIR/conv31_128_2016_11_11_336_1_1_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 336 21 21 2016 1 1 2>&1 |tee $LOG_DIR/conv32_128_2016_21_21_336_1_1_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 336 42 42 1008 1 1 2>&1 |tee $LOG_DIR/conv33_128_1008_42_42_336_1_1_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 64 56 56 64 1 1 2>&1 |tee $LOG_DIR/conv34_128_64_56_56_64_1_1_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S2D1P0 --shape 128 64 112 112 3 7 7 2>&1 |tee $LOG_DIR/conv35_128_3_230_230_64_7_7_2_1_VALID.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S2D1P0 --shape 128 96 165 165 3 3 3 2>&1 |tee $LOG_DIR/conv36_128_3_331_331_96_3_3_2_1_VALID.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 128 56 56 256 1 1 2>&1 |tee $LOG_DIR/conv37_128_256_56_56_128_1_1_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 256 14 14 256 3 3 2>&1 |tee $LOG_DIR/conv38_128_256_14_14_256_3_3_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 672 11 11 2688 1 1 2>&1 |tee $LOG_DIR/conv39_128_2688_11_11_672_1_1_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 168 42 42 1008 1 1 2>&1 |tee $LOG_DIR/conv40_128_1008_42_42_168_1_1_1_1_SAME.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 42 83 83 96 1 1 2>&1 |tee $LOG_DIR/conv41_128_96_83_83_42_1_1_1_1_VALID.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S2D1P0 --shape 128 512 28 28 256 1 1 2>&1 |tee $LOG_DIR/conv42_128_256_56_56_512_1_1_2_1_VALID.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --shape 128 336 21 21 1344 1 1 2>&1 |tee $LOG_DIR/conv43_128_1344_21_21_336_1_1_1_1_SAME.log
