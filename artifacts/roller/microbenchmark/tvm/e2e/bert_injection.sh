PARSE=/home/shanbinke/nnfusion/src/tools/nnfusion/kernel_db/parse_code.py
INJ=/home/shanbinke/nnfusion/src/tools/nnfusion/kernel_db/convert_external.py

if [ $1 == "ansor" ]; then
  prefix="ansor_"
else
  prefix=""
fi

python3 ${PARSE} --op_type Dot --source_file $1/${prefix}matmul_65536_30522_1024.cc --input0_shape 65536 30522 --input1_shape 30522 1024 --output0_shape 65536 1024 --transpose_A False --transpose_B False --json_file=$1/${prefix}matmul_65536_30522_1024.json
python3 ${PARSE} --op_type Dot --source_file $1/${prefix}matmul_65536_2_1024.cc --input0_shape 65536 2 --input1_shape 2 1024 --output0_shape 65536 1024 --transpose_A False --transpose_B False --json_file=$1/${prefix}matmul_65536_2_1024.json
python3 ${PARSE} --op_type Dot --source_file $1/${prefix}matmul_65536_1024_1024.cc --input0_shape 65536 1024 --input1_shape 1024 1024 --output0_shape 65536 1024 --transpose_A False --transpose_B False --json_file=$1/${prefix}matmul_65536_1024_1024.json
python3 ${PARSE} --op_type Dot --source_file $1/${prefix}matmul_65536_1024_4096.cc --input0_shape 65536 1024 --input1_shape 1024 4096 --output0_shape 65536 4096 --transpose_A False --transpose_B False --json_file=$1/${prefix}matmul_65536_1024_4096.json
python3 ${PARSE} --op_type Dot --source_file $1/${prefix}matmul_65536_4096_1024.cc --input0_shape 65536 4096 --input1_shape 4096 1024 --output0_shape 65536 1024 --transpose_A False --transpose_B False --json_file=$1/${prefix}matmul_65536_4096_1024.json
python3 ${PARSE} --op_type BatchMatMul --source_file $1/${prefix}batch_matmul_128_16_512_512_64.cc --input0_shape 128 16 512 512 --input1_shape 128 16 512 64 --output0_shape 128 16 512 64 --transpose_A False --transpose_B False --json_file=$1/${prefix}batch_matmul_128_16_512_512_64.json
python3 ${PARSE} --op_type BatchMatMul --source_file $1/${prefix}batch_matmul_128_16_512_64_512.cc --input0_shape 128 16 512 64 --input1_shape 128 16 64 512 --output0_shape 128 16 512 512 --transpose_A False --transpose_B False --json_file=$1/${prefix}batch_matmul_128_16_512_64_512.json

python3 ${INJ} $1/${prefix}matmul_65536_30522_1024.json
python3 ${INJ} $1/${prefix}matmul_65536_2_1024.json
python3 ${INJ} $1/${prefix}matmul_65536_1024_1024.json
python3 ${INJ} $1/${prefix}matmul_65536_1024_4096.json
python3 ${INJ} $1/${prefix}matmul_65536_4096_1024.json
