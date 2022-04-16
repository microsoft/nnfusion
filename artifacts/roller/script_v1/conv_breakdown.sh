# training bs 128
mkdir -p ./logs/roller/wo_storage_align/conv_breakdown
LOG_DIR=./logs/roller/wo_storage_align/conv_breakdown
CODE_DIR=.

# Roller-B
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S1D1P0 --padding_threshold_cap 0.2 --shape 128 42 83 83 42 1 1 2>&1 |tee $LOG_DIR/conv9_128_42_83_83_42_1_1_1_1_VALID_B.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S1D1P0 --padding_threshold_cap 0.2 --shape 128 42 83 83 96 1 1 2>&1 |tee $LOG_DIR/conv12_128_96_83_83_42_1_1_1_1_SAME_B.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S1D1P0 --padding_threshold_cap 0.2 --shape 128 42 165 165 96 1 1 2>&1 |tee $LOG_DIR/conv13_128_96_165_165_42_1_1_1_1_SAME_B.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S1D1P0 --padding_threshold_cap 0.2 --shape 128 336 21 21 336 1 1 2>&1 |tee $LOG_DIR/conv15_128_336_21_21_336_1_1_1_1_VALID_B.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S1D1P0 --padding_threshold_cap 0.2 --shape 128 168 42 42 336 1 1 2>&1 |tee $LOG_DIR/conv28_128_336_42_42_168_1_1_1_1_SAME_B.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S2D1P0 --padding_threshold_cap 0.2 --shape 128 96 165 165 3 3 3 2>&1 |tee $LOG_DIR/conv36_128_3_331_331_96_3_3_2_1_VALID_B.log

# Roller-F
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --padding_threshold_cap 0.2 --shape 128 42 83 83 42 1 1 2>&1 |tee $LOG_DIR/conv9_128_42_83_83_42_1_1_1_1_VALID_F.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --padding_threshold_cap 0.2 --shape 128 42 83 83 96 1 1 2>&1 |tee $LOG_DIR/conv12_128_96_83_83_42_1_1_1_1_SAME_F.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --padding_threshold_cap 0.2 --shape 128 42 165 165 96 1 1 2>&1 |tee $LOG_DIR/conv13_128_96_165_165_42_1_1_1_1_SAME_F.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --padding_threshold_cap 0.2 --shape 128 336 21 21 336 1 1 2>&1 |tee $LOG_DIR/conv15_128_336_21_21_336_1_1_1_1_VALID_F.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --padding_threshold_cap 0.2 --shape 128 168 42 42 336 1 1 2>&1 |tee $LOG_DIR/conv28_128_336_42_42_168_1_1_1_1_SAME_F.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S2D1P0 --padding_threshold_cap 0.2 --shape 128 96 165 165 3 3 3 2>&1 |tee $LOG_DIR/conv36_128_3_331_331_96_3_3_2_1_VALID_F.log

# Roller-P0.4
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --padding_threshold_cap 0.4 --shape 128 42 83 83 42 1 1 2>&1 |tee $LOG_DIR/conv9_128_42_83_83_42_1_1_1_1_VALID_P0.4.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --padding_threshold_cap 0.4 --shape 128 42 83 83 96 1 1 2>&1 |tee $LOG_DIR/conv12_128_96_83_83_42_1_1_1_1_SAME_P0.4.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --padding_threshold_cap 0.4 --shape 128 42 165 165 96 1 1 2>&1 |tee $LOG_DIR/conv13_128_96_165_165_42_1_1_1_1_SAME_P0.4.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --padding_threshold_cap 0.4 --shape 128 336 21 21 336 1 1 2>&1 |tee $LOG_DIR/conv15_128_336_21_21_336_1_1_1_1_VALID_P0.4.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --padding_threshold_cap 0.4 --shape 128 168 42 42 336 1 1 2>&1 |tee $LOG_DIR/conv28_128_336_42_42_168_1_1_1_1_SAME_P0.4.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S2D1P0 --padding_threshold_cap 0.4 --shape 128 96 165 165 3 3 3 2>&1 |tee $LOG_DIR/conv36_128_3_331_331_96_3_3_2_1_VALID_P0.4.log

# Roller-P1.0
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --padding_threshold_cap 1.0 --shape 128 42 83 83 42 1 1 2>&1 |tee $LOG_DIR/conv9_128_42_83_83_42_1_1_1_1_VALID_P1.0.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --padding_threshold_cap 1.0 --shape 128 42 83 83 96 1 1 2>&1 |tee $LOG_DIR/conv12_128_96_83_83_42_1_1_1_1_SAME_P1.0.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --padding_threshold_cap 1.0 --shape 128 42 165 165 96 1 1 2>&1 |tee $LOG_DIR/conv13_128_96_165_165_42_1_1_1_1_SAME_P1.0.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --padding_threshold_cap 1.0 --shape 128 336 21 21 336 1 1 2>&1 |tee $LOG_DIR/conv15_128_336_21_21_336_1_1_1_1_VALID_P1.0.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P0 --padding_threshold_cap 1.0 --shape 128 168 42 42 336 1 1 2>&1 |tee $LOG_DIR/conv28_128_336_42_42_168_1_1_1_1_SAME_P1.0.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S2D1P0 --padding_threshold_cap 1.0 --shape 128 96 165 165 3 3 3 2>&1 |tee $LOG_DIR/conv36_128_3_331_331_96_3_3_2_1_VALID_P1.0.log
