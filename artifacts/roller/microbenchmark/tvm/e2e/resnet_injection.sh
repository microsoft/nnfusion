PARSE=/home/shanbinke/nnfusion/src/tools/nnfusion/kernel_db/parse_code.py
INJ=/home/shanbinke/nnfusion/src/tools/nnfusion/kernel_db/convert_external.py

if [ $1 == "ansor" ]; then
  prefix="ansor_"
else
  prefix=""
fi

python3 ${PARSE} --op_type Fused_Convolution_Add_Relu --source_file ansor_conv2d_128_3_230_230_64_7_7_2_VALID_add_relu.cc --input0_shape 128 3 230 230 --input1_shape 64 3 7 7 --output0_shape 128 64 112 112 --stride 2 2 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_3_230_230_64_7_7_2_VALID_add_relu.json
python3 ${PARSE} --op_type Fused_Convolution_Add --source_file ansor_conv2d_128_64_56_56_256_1_1_1_SAME_add.cc --input0_shape 128 64 56 56 --input1_shape 256 64 1 1 --output0_shape 128 256 56 56 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_64_56_56_256_1_1_1_SAME_add.json
python3 ${PARSE} --op_type Fused_Convolution_Add_Relu --source_file ansor_conv2d_128_64_56_56_64_1_1_1_SAME_add_relu.cc --input0_shape 128 64 56 56 --input1_shape 64 64 1 1 --output0_shape 128 64 56 56 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_64_56_56_64_1_1_1_SAME_add_relu.json
python3 ${PARSE} --op_type Fused_Convolution_Add_Relu --source_file ansor_conv2d_128_64_56_56_64_3_3_1_SAME_add_relu.cc --input0_shape 128 64 56 56 --input1_shape 64 64 3 3 --output0_shape 128 64 56 56 --stride 1 1 --padding 1 1 --dilation 1 1 --json_file=ansor_conv2d_128_64_56_56_64_3_3_1_SAME_add_relu.json
python3 ${PARSE} --op_type Fused_Convolution_Add_Relu --source_file ansor_conv2d_128_256_56_56_64_1_1_1_SAME_add_relu.cc --input0_shape 128 256 56 56 --input1_shape 64 256 1 1 --output0_shape 128 64 56 56 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_256_56_56_64_1_1_1_SAME_add_relu.json
python3 ${PARSE} --op_type Fused_Convolution_Add --source_file ansor_conv2d_128_256_56_56_512_1_1_2_VALID_add.cc --input0_shape 128 256 56 56 --input1_shape 512 256 1 1 --output0_shape 128 512 28 28 --stride 2 2 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_256_56_56_512_1_1_2_VALID_add.json
python3 ${PARSE} --op_type Fused_Convolution_Add_Relu --source_file ansor_conv2d_128_256_56_56_128_1_1_1_SAME_add_relu.cc --input0_shape 128 256 56 56 --input1_shape 128 256 1 1 --output0_shape 128 128 56 56 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_256_56_56_128_1_1_1_SAME_add_relu.json
python3 ${PARSE} --op_type Fused_Convolution_Add_Relu --source_file ansor_conv2d_128_128_58_58_128_3_3_2_VALID_add_relu.cc --input0_shape 128 128 58 58 --input1_shape 128 128 3 3 --output0_shape 128 128 28 28 --stride 2 2 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_128_58_58_128_3_3_2_VALID_add_relu.json
python3 ${PARSE} --op_type Fused_Convolution_Add --source_file ansor_conv2d_128_128_28_28_512_1_1_1_SAME_add.cc --input0_shape 128 128 28 28 --input1_shape 512 128 1 1 --output0_shape 128 512 28 28 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_128_28_28_512_1_1_1_SAME_add.json
python3 ${PARSE} --op_type Fused_Convolution_Add_Relu --source_file ansor_conv2d_128_512_28_28_128_1_1_1_SAME_add_relu.cc --input0_shape 128 512 28 28 --input1_shape 128 512 1 1 --output0_shape 128 128 28 28 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_512_28_28_128_1_1_1_SAME_add_relu.json
python3 ${PARSE} --op_type Fused_Convolution_Add_Relu --source_file ansor_conv2d_128_128_28_28_128_3_3_1_SAME_add_relu.cc --input0_shape 128 128 28 28 --input1_shape 128 128 3 3 --output0_shape 128 128 28 28 --stride 1 1 --padding 1 1 --dilation 1 1 --json_file=ansor_conv2d_128_128_28_28_128_3_3_1_SAME_add_relu.json
python3 ${PARSE} --op_type Fused_Convolution_Add --source_file ansor_conv2d_128_512_28_28_1024_1_1_2_VALID_add.cc --input0_shape 128 512 28 28 --input1_shape 1024 512 1 1 --output0_shape 128 1024 14 14 --stride 2 2 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_512_28_28_1024_1_1_2_VALID_add.json
python3 ${PARSE} --op_type Fused_Convolution_Add_Relu --source_file ansor_conv2d_128_512_28_28_256_1_1_1_SAME_add_relu.cc --input0_shape 128 512 28 28 --input1_shape 256 512 1 1 --output0_shape 128 256 28 28 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_512_28_28_256_1_1_1_SAME_add_relu.json
python3 ${PARSE} --op_type Fused_Convolution_Add_Relu --source_file ansor_conv2d_128_256_30_30_256_3_3_2_VALID_add_relu.cc --input0_shape 128 256 30 30 --input1_shape 256 256 3 3 --output0_shape 128 256 14 14 --stride 2 2 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_256_30_30_256_3_3_2_VALID_add_relu.json
python3 ${PARSE} --op_type Fused_Convolution_Add --source_file ansor_conv2d_128_256_14_14_1024_1_1_1_SAME_add.cc --input0_shape 128 256 14 14 --input1_shape 1024 256 1 1 --output0_shape 128 1024 14 14 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_256_14_14_1024_1_1_1_SAME_add.json
python3 ${PARSE} --op_type Fused_Convolution_Add_Relu --source_file ansor_conv2d_128_1024_14_14_256_1_1_1_SAME_add_relu.cc --input0_shape 128 1024 14 14 --input1_shape 256 1024 1 1 --output0_shape 128 256 14 14 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_1024_14_14_256_1_1_1_SAME_add_relu.json
python3 ${PARSE} --op_type Fused_Convolution_Add_Relu --source_file ansor_conv2d_128_256_14_14_256_3_3_1_SAME_add_relu.cc --input0_shape 128 256 14 14 --input1_shape 256 256 3 3 --output0_shape 128 256 14 14 --stride 1 1 --padding 1 1 --dilation 1 1 --json_file=ansor_conv2d_128_256_14_14_256_3_3_1_SAME_add_relu.json
python3 ${PARSE} --op_type Fused_Convolution_Add --source_file ansor_conv2d_128_1024_14_14_2048_1_1_2_VALID_add.cc --input0_shape 128 1024 14 14 --input1_shape 2048 1024 1 1 --output0_shape 128 2048 7 7 --stride 2 2 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_1024_14_14_2048_1_1_2_VALID_add.json
python3 ${PARSE} --op_type Fused_Convolution_Add_Relu --source_file ansor_conv2d_128_1024_14_14_512_1_1_1_SAME_add_relu.cc --input0_shape 128 1024 14 14 --input1_shape 512 1024 1 1 --output0_shape 128 512 14 14 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_1024_14_14_512_1_1_1_SAME_add_relu.json
python3 ${PARSE} --op_type Fused_Convolution_Add_Relu --source_file ansor_conv2d_128_512_16_16_512_3_3_2_VALID_add_relu.cc --input0_shape 128 512 16 16 --input1_shape 512 512 3 3 --output0_shape 128 512 7 7 --stride 2 2 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_512_16_16_512_3_3_2_VALID_add_relu.json
python3 ${PARSE} --op_type Fused_Convolution_Add --source_file ansor_conv2d_128_512_7_7_2048_1_1_1_SAME_add.cc --input0_shape 128 512 7 7 --input1_shape 2048 512 1 1 --output0_shape 128 2048 7 7 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_512_7_7_2048_1_1_1_SAME_add.json
python3 ${PARSE} --op_type Fused_Convolution_Add_Relu --source_file ansor_conv2d_128_2048_7_7_512_1_1_1_SAME_add_relu.cc --input0_shape 128 2048 7 7 --input1_shape 512 2048 1 1 --output0_shape 128 512 7 7 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_2048_7_7_512_1_1_1_SAME_add_relu.json
python3 ${PARSE} --op_type Fused_Convolution_Add_Relu --source_file ansor_conv2d_128_512_7_7_512_3_3_1_SAME_add_relu.cc --input0_shape 128 512 7 7 --input1_shape 512 512 3 3 --output0_shape 128 512 7 7 --stride 1 1 --padding 1 1 --dilation 1 1 --json_file=ansor_conv2d_128_512_7_7_512_3_3_1_SAME_add_relu.json
python3 ${PARSE} --op_type Dot --source_file $1/${prefix}matmul_128_2048_1000.cc --input0_shape 128 2048 --input1_shape 2048 1000 --output0_shape 128 1000 --transpose_A False --transpose_B False --json_file=$1/${prefix}matmul_128_2048_1000.json

python3 ${INJ} ansor_conv2d_128_3_230_230_64_7_7_2_VALID_add_relu.json
python3 ${INJ} ansor_conv2d_128_64_56_56_256_1_1_1_SAME_add.json
python3 ${INJ} ansor_conv2d_128_64_56_56_64_1_1_1_SAME_add_relu.json
python3 ${INJ} ansor_conv2d_128_64_56_56_64_3_3_1_SAME_add_relu.json
python3 ${INJ} ansor_conv2d_128_256_56_56_64_1_1_1_SAME_add_relu.json
python3 ${INJ} ansor_conv2d_128_256_56_56_512_1_1_2_VALID_add.json
python3 ${INJ} ansor_conv2d_128_256_56_56_128_1_1_1_SAME_add_relu.json
python3 ${INJ} ansor_conv2d_128_128_58_58_128_3_3_2_VALID_add_relu.json
python3 ${INJ} ansor_conv2d_128_128_28_28_512_1_1_1_SAME_add.json
python3 ${INJ} ansor_conv2d_128_512_28_28_128_1_1_1_SAME_add_relu.json
python3 ${INJ} ansor_conv2d_128_128_28_28_128_3_3_1_SAME_add_relu.json
python3 ${INJ} ansor_conv2d_128_512_28_28_1024_1_1_2_VALID_add.json
python3 ${INJ} ansor_conv2d_128_512_28_28_256_1_1_1_SAME_add_relu.json
python3 ${INJ} ansor_conv2d_128_256_30_30_256_3_3_2_VALID_add_relu.json
python3 ${INJ} ansor_conv2d_128_256_14_14_1024_1_1_1_SAME_add.json
python3 ${INJ} ansor_conv2d_128_1024_14_14_256_1_1_1_SAME_add_relu.json
python3 ${INJ} ansor_conv2d_128_256_14_14_256_3_3_1_SAME_add_relu.json
python3 ${INJ} ansor_conv2d_128_1024_14_14_2048_1_1_2_VALID_add.json
python3 ${INJ} ansor_conv2d_128_1024_14_14_512_1_1_1_SAME_add_relu.json
python3 ${INJ} ansor_conv2d_128_512_16_16_512_3_3_2_VALID_add_relu.json
python3 ${INJ} ansor_conv2d_128_512_7_7_2048_1_1_1_SAME_add.json
python3 ${INJ} ansor_conv2d_128_2048_7_7_512_1_1_1_SAME_add_relu.json
python3 ${INJ} ansor_conv2d_128_512_7_7_512_3_3_1_SAME_add_relu.json
python3 ${INJ} $1/${prefix}matmul_128_2048_1000.json
