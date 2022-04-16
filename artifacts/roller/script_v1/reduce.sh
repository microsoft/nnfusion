# training bs 128
mkdir -p ./logs/roller/wo_storage_align/reduce
LOG_DIR=./logs/roller/wo_storage_align/reduce
CODE_DIR=.

python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/reduce --smem_tiling --reg_tiling --op reduce_expr2 --shape 65536 1024 2>&1 | tee $LOG_DIR/reduce0_128_512_1024.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/reduce --smem_tiling --reg_tiling --op reduce_expr2 --shape 65536 1024 2>&1 | tee $LOG_DIR/reduce1_65536_1024.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/reduce --smem_tiling --reg_tiling --op reduce_expr2 --shape 516096 121 2>&1 | tee $LOG_DIR/reduce2_128_4032_11_11.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/reduce --smem_tiling --reg_tiling --op reduce_expr2 --shape 262144 49 2>&1 | tee $LOG_DIR/reduce3_128_2048_7_7.log
