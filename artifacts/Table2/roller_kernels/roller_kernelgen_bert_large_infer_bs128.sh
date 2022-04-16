# kernel generation for nasnet large batch 128

rm -rf bert_kernels
mkdir -p bert_kernels
mkdir -p bert_kernels/src
mkdir -p bert_kernels/json

ROLLER_HOME=../../roller
ROLLER_KERNEL_DIR=$PWD
cd $ROLLER_HOME

# batch matmul
python test_op_mp.py --code_dir $ROLLER_KERNEL_DIR/bert_kernels/src --smem_tiling --reg_tiling --op batch_matmul_expr --shape 2048 512 512 64 2>&1
python test_op_mp.py --code_dir $ROLLER_KERNEL_DIR/bert_kernels/src --smem_tiling --reg_tiling --op batch_matmul_expr --shape 2048 512 64 512 2>&1

# broadcast
# TODO!!!
# python test_op_mp.py --code_dir $ROLLER_KERNEL_DIR/bert_kernels/src --smem_tiling --reg_tiling --op broadcast_expr --shape 128 512 512 1 2>&1
# python test_op_mp.py --code_dir $ROLLER_KERNEL_DIR/bert_kernels/src --smem_tiling --reg_tiling --op broadcast_expr --shape 128 16 512 512 1 2>&1

# Dot
python test_op_mp.py --code_dir $ROLLER_KERNEL_DIR/bert_kernels/src --smem_tiling --reg_tiling --op matmul_expr --shape 65536 1024 30522 2>&1
python test_op_mp.py --code_dir $ROLLER_KERNEL_DIR/bert_kernels/src --smem_tiling --reg_tiling --op matmul_expr --shape 65536 1024 2 2>&1
python test_op_mp.py --code_dir $ROLLER_KERNEL_DIR/bert_kernels/src --smem_tiling --reg_tiling --op matmul_expr --shape 65536 1024 1024 2>&1
python test_op_mp.py --code_dir $ROLLER_KERNEL_DIR/bert_kernels/src --smem_tiling --reg_tiling --op matmul_expr --shape 65536 4096 1024 2>&1
python test_op_mp.py --code_dir $ROLLER_KERNEL_DIR/bert_kernels/src --smem_tiling --reg_tiling --op matmul_expr --shape 65536 1024 4096 2>&1

# Sum
python test_op_mp.py --code_dir $ROLLER_KERNEL_DIR/bert_kernels/src --smem_tiling --reg_tiling --op reduce_expr3 --shape 128 512 1024 1 2>&1
python test_op_mp.py --code_dir $ROLLER_KERNEL_DIR/bert_kernels/src --smem_tiling --reg_tiling --op reduce_expr1 --shape 65536 1024 1 2>&1

# kernel code to json
cd $ROLLER_KERNEL_DIR

python parse_code.py --op_type BatchMatMul --source_file ./bert_kernels/src/roller_batch_matmul_expr_2048_512_64_512.cu --json_file bert_kernels/json/roller_BatchMatMul_[128,16,512,512]_[128,16,512,64]_[128,16,512,64].json --input0_shape 128 16 512 512 --input1_shape 128 16 512 64 --output0_shape 128 16 512 64
python parse_code.py --op_type BatchMatMul --source_file ./bert_kernels/src/roller_batch_matmul_expr_2048_512_512_64.cu --json_file bert_kernels/json/roller_BatchMatMul_[128,16,512,64]_[128,16,512,64]_[128,16,512,512].json --input0_shape 128 16 512 64 --input1_shape 128 16 64 512 --output0_shape 128 16 512 512
# python parse_code.py --op_type Broadcast --source_file ./bert_kernels/src/roller_Broadcast_\[128\,512\]_\[128\,512\,512\].cu --json_file bert_kernels/json/roller_Broadcast_\[128\,512\]_\[128\,512\,512\].json --input0_shape 128 512 --output0_shape 128 512 512 --broadcast_axis 1
# python parse_code.py --op_type Broadcast --source_file ./bert_kernels/src/roller_Broadcast_\[128\,512\,512\]_\[128\,16\,512\,512\].cu --json_file bert_kernels/json/roller_Broadcast_\[128\,512\,512\]_\[128\,16\,512\,512\].json --input0_shape 128 512 512 --output0_shape 128 16 512 512 --broadcast_axis 1
python parse_code.py --op_type Dot --source_file ./bert_kernels/src/roller_matmul_expr_65536_1024_1024.cu --json_file bert_kernels/json/roller_Dot_\[65536\,1024\]_\[1024\,1024\]_\[65536\,1024\].json --input0_shape 65536 1024 --input1_shape 1024 1024 --output0_shape 65536 1024
python parse_code.py --op_type Dot --source_file ./bert_kernels/src/roller_matmul_expr_65536_4096_1024.cu --json_file bert_kernels/json/roller_Dot_\[65536\,1024\]_\[1024\,4096\]_\[65536\,1024\].json --input0_shape 65536 1024 --input1_shape 1024 4096 --output0_shape 65536 4096
python parse_code.py --op_type Dot --source_file ./bert_kernels/src/roller_matmul_expr_65536_1024_2.cu --json_file bert_kernels/json/roller_Dot_\[65536\,2\]_\[2\,1024\]_\[65536\,1024\].json --input0_shape 65536 2 --input1_shape 2 1024 --output0_shape 65536 1024
python parse_code.py --op_type Dot --source_file ./bert_kernels/src/roller_matmul_expr_65536_1024_30522.cu --json_file bert_kernels/json/roller_Dot_\[65536\,30522\]_\[30522\,1024\]_\[65536\,1024\].json --input0_shape 65536 30522 --input1_shape 30522 1024 --output0_shape 65536 1024
python parse_code.py --op_type Dot --source_file ./bert_kernels/src/roller_matmul_expr_65536_1024_4096.cu --json_file bert_kernels/json/roller_Dot_\[65536\,4096\]_\[4096\,1024\]_\[65536\,1024\].json --input0_shape 65536 4096 --input1_shape 4096 1024 --output0_shape 65536 1024
python parse_code.py --op_type Sum --source_file ./bert_kernels/src/roller_reduce_expr3_128_512_1024_1.cu --json_file bert_kernels/json/roller_Sum_\[128\,512\,1024\]_\[128\,512\].json --input0_shape 128 512 1024 --output0_shape 128 512 --reduction_axis 2
python parse_code.py --op_type Sum --source_file ./bert_kernels/src/roller_reduce_expr1_65536_1024_1.cu --json_file bert_kernels/json/roller_Sum_\[65536\,1024\]_\[65536\].json --input0_shape 65536 1024 --output0_shape 65536 --reduction_axis 1

# json file to kernel_db
python convert_external.py --json_file ./bert_kernels/json/roller_BatchMatMul_[128,16,512,512]_[128,16,512,64]_[128,16,512,64].json --db_path ./bert_kernels/
python convert_external.py --json_file ./bert_kernels/json/roller_BatchMatMul_[128,16,512,64]_[128,16,512,64]_[128,16,512,512].json --db_path ./bert_kernels/
# python convert_external.py --json_file ./bert_kernels/json/roller_Broadcast_[128,512]_[128,512,512].json --db_path ./bert_kernels/
# python convert_external.py --json_file ./bert_kernels/json/roller_Broadcast_[128,512,512]_[128,16,512,512].json --db_path ./bert_kernels/
python convert_external.py --json_file ./bert_kernels/json/roller_Dot_[65536,1024]_[1024,1024]_[65536,1024].json --db_path ./bert_kernels/
python convert_external.py --json_file ./bert_kernels/json/roller_Dot_[65536,1024]_[1024,4096]_[65536,1024].json --db_path ./bert_kernels/
python convert_external.py --json_file ./bert_kernels/json/roller_Dot_[65536,2]_[2,1024]_[65536,1024].json --db_path ./bert_kernels/
python convert_external.py --json_file ./bert_kernels/json/roller_Dot_[65536,30522]_[30522,1024]_[65536,1024].json --db_path ./bert_kernels/
python convert_external.py --json_file ./bert_kernels/json/roller_Dot_[65536,4096]_[4096,1024]_[65536,1024].json --db_path ./bert_kernels/
python convert_external.py --json_file ./bert_kernels/json/roller_Sum_[128,512,1024]_[128,512].json --db_path ./bert_kernels/
python convert_external.py --json_file ./bert_kernels/json/roller_Sum_[65536,1024]_[65536].json --db_path ./bert_kernels/
