# training bs 128
mkdir -p ./logs/roller/wo_storage_align/tc_matmul
LOG_DIR=./logs/roller/wo_storage_align/tc_matmul
CODE_DIR=.

python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/tc_matmul --smem_tiling --reg_tiling --op matmul_expr --use_tc --data_type float16 --shape 65536 4096 1024 2>&1 | tee $LOG_DIR/tc_mm0_65536_4096_1024.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/tc_matmul --smem_tiling --reg_tiling --op matmul_expr --use_tc --data_type float16 --shape 65536 1024 1024 2>&1 | tee $LOG_DIR/tc_mm1_65536_1024_1024.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/tc_matmul --smem_tiling --reg_tiling --op matmul_expr --use_tc --data_type float16 --shape 65536 1024 4096 2>&1 | tee $LOG_DIR/tc_mm2_65536_1024_4096.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/tc_matmul --smem_tiling --reg_tiling --op matmul_expr --use_tc --data_type float16 --shape 65536 1024 16384 2>&1 | tee $LOG_DIR/tc_mm3_65536_1024_16384.log
