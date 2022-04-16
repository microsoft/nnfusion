# kernel generation for nasnet large batch 128

rm -rf lstm_kernels
mkdir -p lstm_kernels
mkdir -p lstm_kernels/src
mkdir -p lstm_kernels/json

ROLLER_HOME=../../roller
ROLLER_KERNEL_DIR=$PWD
cd $ROLLER_HOME

# Dot
python test_op_mp.py --keep_tiny --code_dir $ROLLER_KERNEL_DIR/lstm_kernels/src --smem_tiling --reg_tiling --op matmul_expr --shape 128 256 256 2>&1

# kernel code to json
cd $ROLLER_KERNEL_DIR

python parse_code.py --op_type Dot --source_file ./lstm_kernels/src/roller_matmul_expr_128_256_256.cu --json_file ./lstm_kernels/json/roller_Dot_\[128\,256\]_\[256\,256\]_\[256\,256\].json --input0_shape 128 256 --input1_shape 256 256 --output0_shape 128 256

# json file to kernel_db
python convert_external.py --json_file ./lstm_kernels/json/roller_Dot_\[128\,256\]_\[256\,256\]_\[256\,256\].json --db_path ./lstm_kernels/
