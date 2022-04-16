# training bs 128
mkdir -p ./logs_v2_rocm/roller/wo_storage_align/matmul
LOG_DIR=./logs_v2_rocm/roller/wo_storage_align/matmul
CODE_DIR=.

python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir matmul --smem_tiling --reg_tiling --op matmul_expr --shape 65536 1024 2 2>&1 | tee $LOG_DIR/matmul0_65536_1024_2.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir matmul --smem_tiling --reg_tiling --op matmul_expr --shape 128 1000 4032 2>&1 | tee $LOG_DIR/matmul1_128_1000_4032.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir matmul --smem_tiling --reg_tiling --op matmul_expr --shape 128 1000 2048 2>&1 | tee $LOG_DIR/matmul2_128_1000_2048.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir matmul --smem_tiling --reg_tiling --op matmul_expr --shape 65536 4096 1024 2>&1 | tee $LOG_DIR/matmul3_65536_4096_1024.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir matmul --smem_tiling --reg_tiling --op matmul_expr --shape 65536 1024 1024 2>&1 | tee $LOG_DIR/matmul4_65536_1024_1024.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir matmul --smem_tiling --reg_tiling --op matmul_expr --shape 65536 1024 4096 2>&1 | tee $LOG_DIR/matmul5_65536_1024_4096.log
python3 -u $CODE_DIR/test_op_rocm_mp.py --code_dir matmul --smem_tiling --reg_tiling --op matmul_expr --shape 65536 1024 30522 2>&1 | tee $LOG_DIR/matmul6_65536_1024_30522.log

