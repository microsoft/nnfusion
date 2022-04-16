PARSE=/home/shanbinke/nnfusion/src/tools/nnfusion/kernel_db/parse_code.py
INJ=/home/shanbinke/nnfusion/src/tools/nnfusion/kernel_db/convert_external.py

if [ $1 == "ansor" ]; then
  prefix="ansor_"
else
  prefix=""
fi

python3 ${PARSE} --op_type Dot --source_file $1/${prefix}matmul_128_256_256.cc --input0_shape 128 256 --input1_shape 256 256 --output0_shape 128 256 --transpose_A False --transpose_B False --json_file=$1/${prefix}matmul_128_256_256.json

python3 ${INJ} $1/${prefix}matmul_128_256_256.json
