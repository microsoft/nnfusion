mkdir -p ./logs/roller/wo_storage_align/tiny_matmul
LOG_DIR=./logs/roller/wo_storage_align/tiny_matmul
CODE_DIR=.

# Roller-S
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/matmul --smem_tiling --reg_tiling --op matmul_expr --shape 128 1000 4032 2>&1 | tee $LOG_DIR/matmul1_128_1000_4032_S.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/matmul --smem_tiling --reg_tiling --op matmul_expr --shape 128 1000 2048 2>&1 | tee $LOG_DIR/matmul2_128_1000_2048_S.log

# Roller-O
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/matmul --smem_tiling --reg_tiling --op matmul_expr --keep_tiny --shape 128 1000 4032 2>&1 | tee $LOG_DIR/matmul1_128_1000_4032_O.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/matmul --smem_tiling --reg_tiling --op matmul_expr --keep_tiny --shape 128 1000 2048 2>&1 | tee $LOG_DIR/matmul2_128_1000_2048_O.log
