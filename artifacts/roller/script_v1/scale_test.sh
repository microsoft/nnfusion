mkdir -p ./logs/roller/wo_storage_align/scale/relu
LOG_DIR=./logs/roller/wo_storage_align/scale/relu
CODE_DIR=.

python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 1048576 2>&1 |tee $LOG_DIR/relu0_1048576.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 2097152 2>&1 |tee $LOG_DIR/relu1_2097152.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 4194304 2>&1 |tee $LOG_DIR/relu2_4194304.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 8388608 2>&1 |tee $LOG_DIR/relu3_8388608.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 16777216 2>&1 |tee $LOG_DIR/relu4_16777216.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 33554432 2>&1 |tee $LOG_DIR/relu5_33554432.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 67108864 2>&1 |tee $LOG_DIR/relu6_67108864.log

mkdir -p ./logs/roller/wo_storage_align/scale/conv
LOG_DIR=./logs/roller/wo_storage_align/scale/conv
CODE_DIR=.

python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S2D1P0 --shape 128 2048 7 7 1024 1 1 2>&1 |tee $LOG_DIR/conv0_128_1024_14_14_2048_1_1_2_1_VALID.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S2D1P0 --shape 256 2048 7 7 1024 1 1 2>&1 |tee $LOG_DIR/conv1_256_1024_14_14_2048_1_1_2_1_VALID.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S2D1P0 --shape 512 2048 7 7 1024 1 1 2>&1 |tee $LOG_DIR/conv2_512_1024_14_14_2048_1_1_2_1_VALID.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S2D1P0 --shape 1024 2048 7 7 1024 1 1 2>&1 |tee $LOG_DIR/conv3_1024_1024_14_14_2048_1_1_2_1_VALID.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S2D1P0 --shape 2048 2048 7 7 1024 1 1 2>&1 |tee $LOG_DIR/conv4_2048_1024_14_14_2048_1_1_2_1_VALID.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S2D1P0 --shape 4096 2048 7 7 1024 1 1 2>&1 |tee $LOG_DIR/conv5_4096_1024_14_14_2048_1_1_2_1_VALID.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/conv --schedule_fuse --smem_tiling --reg_tiling --op fused_conv_expr_S2D1P0 --shape 8192 2048 7 7 1024 1 1 2>&1 |tee $LOG_DIR/conv6_8192_1024_14_14_2048_1_1_2_1_VALID.log

mkdir -p ./logs/roller/wo_storage_align/scale/matmul
LOG_DIR=./logs/roller/wo_storage_align/scale/matmul
CODE_DIR=.

python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/matmul --smem_tiling --reg_tiling --op matmul_expr --shape 512 1024 4096 2>&1 | tee $LOG_DIR/matmul0_512_1024_4096.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/matmul --smem_tiling --reg_tiling --op matmul_expr --shape 1024 1024 4096 2>&1 | tee $LOG_DIR/matmul1_1024_1024_4096.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/matmul --smem_tiling --reg_tiling --op matmul_expr --shape 2048 1024 4096 2>&1 | tee $LOG_DIR/matmul2_2048_1024_4096.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/matmul --smem_tiling --reg_tiling --op matmul_expr --shape 4096 1024 4096 2>&1 | tee $LOG_DIR/matmul3_4096_1024_4096.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/matmul --smem_tiling --reg_tiling --op matmul_expr --shape 8192 1024 4096 2>&1 | tee $LOG_DIR/matmul4_8192_1024_4096.log
python3.7 -u $CODE_DIR/test_op_mp.py --code_dir generated_source/matmul --smem_tiling --reg_tiling --op matmul_expr --shape 16384 1024 4096 2>&1 | tee $LOG_DIR/matmul5_16384_1024_4096.log
