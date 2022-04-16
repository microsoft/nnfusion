PARSE=/home/shanbinke/nnfusion/src/tools/nnfusion/kernel_db/parse_code.py
INJ=/home/shanbinke/nnfusion/src/tools/nnfusion/kernel_db/convert_external.py

if [ $1 == "ansor" ]; then
  prefix="ansor_"
else
  prefix=""
fi

python3 ${PARSE} --op_type Fused_Convolution_Add_Relu --source_file ansor_conv2d_128_3_331_331_96_3_3_2_VALID_add_relu.cc --input0_shape 128 3 331 331 --input1_shape 96 3 3 3 --output0_shape 128 96 165 165 --stride 2 2 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_3_331_331_96_3_3_2_VALID_add_relu.json
python3 ${PARSE} --op_type Fused_Convolution_Add --source_file ansor_conv2d_128_96_165_165_42_1_1_1_SAME_add.cc --input0_shape 128 96 165 165 --input1_shape 42 96 1 1 --output0_shape 128 42 165 165 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_96_165_165_42_1_1_1_SAME_add.json
python3 ${PARSE} --op_type Fused_Convolution_Add_Relu --source_file ansor_conv2d_128_96_83_83_42_1_1_1_SAME_add_relu.cc --input0_shape 128 96 83 83 --input1_shape 42 96 1 1 --output0_shape 128 42 83 83 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_96_83_83_42_1_1_1_SAME_add_relu.json
python3 ${PARSE} --op_type Fused_Convolution_Add --source_file ansor_conv2d_128_42_83_83_42_1_1_1_VALID_add.cc --input0_shape 128 42 83 83 --input1_shape 42 42 1 1 --output0_shape 128 42 83 83 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_42_83_83_42_1_1_1_SAME_add.json
python3 ${PARSE} --op_type Fused_Convolution_Add --source_file ansor_conv2d_128_168_83_83_84_1_1_1_SAME_add.cc --input0_shape 128 168 83 83 --input1_shape 84 168 1 1 --output0_shape 128 84 83 83 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_168_83_83_84_1_1_1_VALID_add.json
python3 ${PARSE} --op_type Convolution --source_file ansor_conv2d_128_96_83_83_42_1_1_1_SAME.cc --input0_shape 128 96 83 83 --input1_shape 42 96 1 1 --output0_shape 128 42 83 83 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_96_83_83_42_1_1_1_SAME.json
python3 ${PARSE} --op_type Fused_Convolution_Add_Relu --source_file ansor_conv2d_128_84_42_42_84_1_1_1_VALID_add_relu.cc --input0_shape 128 84 42 42 --input1_shape 84 84 1 1 --output0_shape 128 84 42 42 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_84_42_42_84_1_1_1_VALID_add_relu.json
python3 ${PARSE} --op_type Fused_Convolution_Add --source_file ansor_conv2d_128_336_42_42_168_1_1_1_SAME_add.cc --input0_shape 128 336 42 42 --input1_shape 168 336 1 1 --output0_shape 128 168 42 42 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_336_42_42_168_1_1_1_SAME_add.json
python3 ${PARSE} --op_type Fused_Convolution_Add_Relu --source_file ansor_conv2d_128_168_42_42_168_1_1_1_VALID_add_relu.cc --input0_shape 128 168 42 42 --input1_shape 168 168 1 1 --output0_shape 128 168 42 42 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_168_42_42_168_1_1_1_VALID_add_relu.json
python3 ${PARSE} --op_type Convolution --source_file ansor_conv2d_128_168_42_42_84_1_1_1_SAME.cc --input0_shape 128 168 42 42 --input1_shape 84 168 1 1 --output0_shape 128 84 42 42 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_168_42_42_84_1_1_1_SAME.json
python3 ${PARSE} --op_type Fused_Convolution_Add --source_file ansor_conv2d_128_1008_42_42_168_1_1_1_SAME_add.cc --input0_shape 128 1008 42 42 --input1_shape 168 1008 1 1 --output0_shape 128 168 42 42 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_1008_42_42_168_1_1_1_SAME_add.json
python3 ${PARSE} --op_type Fused_Convolution_Add --source_file ansor_conv2d_128_1008_42_42_336_1_1_1_SAME_add.cc --input0_shape 128 1008 42 42 --input1_shape 336 1008 1 1 --output0_shape 128 336 42 42 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_1008_42_42_336_1_1_1_SAME_add.json
python3 ${PARSE} --op_type Fused_Convolution_Add_Relu --source_file ansor_conv2d_128_336_21_21_336_1_1_1_VALID_add_relu.cc --input0_shape 128 336 21 21 --input1_shape 336 336 1 1 --output0_shape 128 336 21 21 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_336_21_21_336_1_1_1_VALID_add_relu.json
python3 ${PARSE} --op_type Fused_Convolution_Add --source_file ansor_conv2d_128_1344_21_21_336_1_1_1_SAME_add.cc --input0_shape 128 1344 21 21 --input1_shape 336 1344 1 1 --output0_shape 128 336 21 21 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_1344_21_21_336_1_1_1_SAME_add.json
python3 ${PARSE} --op_type Convolution --source_file ansor_conv2d_128_1008_21_21_168_1_1_1_SAME.cc --input0_shape 128 1008 21 21 --input1_shape 168 1008 1 1 --output0_shape 128 168 21 21 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_1008_21_21_168_1_1_1_SAME.json
python3 ${PARSE} --op_type Fused_Convolution_Add --source_file ansor_conv2d_128_2016_21_21_336_1_1_1_SAME_add.cc --input0_shape 128 2016 21 21 --input1_shape 336 2016 1 1 --output0_shape 128 336 21 21 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_2016_21_21_336_1_1_1_SAME_add.json
python3 ${PARSE} --op_type Fused_Convolution_Add --source_file ansor_conv2d_128_2016_21_21_672_1_1_1_SAME_add.cc --input0_shape 128 2016 21 21 --input1_shape 672 2016 1 1 --output0_shape 128 672 21 21 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_2016_21_21_672_1_1_1_SAME_add.json
python3 ${PARSE} --op_type Fused_Convolution_Add_Relu --source_file ansor_conv2d_128_672_11_11_672_1_1_1_VALID_add_relu.cc --input0_shape 128 672 11 11 --input1_shape 672 672 1 1 --output0_shape 128 672 11 11 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_672_11_11_672_1_1_1_VALID_add_relu.json
python3 ${PARSE} --op_type Fused_Convolution_Add --source_file ansor_conv2d_128_2688_11_11_672_1_1_1_SAME_add.cc --input0_shape 128 2688 11 11 --input1_shape 672 2688 1 1 --output0_shape 128 672 11 11 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_2688_11_11_672_1_1_1_SAME_add.json
python3 ${PARSE} --op_type Convolution --source_file ansor_conv2d_128_2016_11_11_336_1_1_1_SAME.cc --input0_shape 128 2016 11 11 --input1_shape 336 2016 1 1 --output0_shape 128 336 11 11 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_2016_11_11_336_1_1_1_SAME.json
python3 ${PARSE} --op_type Fused_Convolution_Add --source_file ansor_conv2d_128_4032_11_11_672_1_1_1_SAME_add.cc --input0_shape 128 4032 11 11 --input1_shape 672 4032 1 1 --output0_shape 128 672 11 11 --stride 1 1 --padding 0 0 --dilation 1 1 --json_file=ansor_conv2d_128_4032_11_11_672_1_1_1_SAME_add.json


python3 ${PARSE} --op_type DepthwiseConv2dNative --source_file $1/${prefix}depthwise_128_96_165_165_7_7_2_SAME.cc --input0_shape 128 96 165 165 --input1_shape 96 1 7 7 --output0_shape 128 96 83 83 --stride 2 2 --padding 3 3 --dilation 1 1 --json_file=$1/${prefix}depthwise_128_96_165_165_7_7_2_SAME.json
python3 ${PARSE} --op_type DepthwiseConv2dNative --source_file $1/${prefix}depthwise_128_42_83_83_7_7_1_SAME.cc --input0_shape 128 42 83 83 --input1_shape 42 1 7 7 --output0_shape 128 42 83 83 --stride 1 1 --padding 3 3 --dilation 1 1 --json_file=$1/${prefix}depthwise_128_42_83_83_7_7_1_SAME.json
python3 ${PARSE} --op_type DepthwiseConv2dNative --source_file $1/${prefix}depthwise_128_42_165_165_5_5_2_SAME.cc --input0_shape 128 42 165 165 --input1_shape 42 1 5 5 --output0_shape 128 42 83 83 --stride 2 2 --padding 2 2 --dilation 1 1 --json_file=$1/${prefix}depthwise_128_42_165_165_5_5_2_SAME.json
python3 ${PARSE} --op_type DepthwiseConv2dNative --source_file $1/${prefix}depthwise_128_42_83_83_5_5_1_SAME.cc --input0_shape 128 42 83 83 --input1_shape 42 1 5 5 --output0_shape 128 42 83 83 --stride 1 1 --padding 2 2 --dilation 1 1 --json_file=$1/${prefix}depthwise_128_42_83_83_5_5_1_SAME.json
python3 ${PARSE} --op_type DepthwiseConv2dNative --source_file $1/${prefix}depthwise_128_42_83_83_3_3_1_SAME.cc --input0_shape 128 42 83 83 --input1_shape 42 1 3 3 --output0_shape 128 42 83 83 --stride 1 1 --padding 1 1 --dilation 1 1 --json_file=$1/${prefix}depthwise_128_42_83_83_3_3_1_SAME.json
python3 ${PARSE} --op_type DepthwiseConv2dNative --source_file $1/${prefix}depthwise_128_96_165_165_5_5_2_SAME.cc --input0_shape 128 96 165 165 --input1_shape 96 1 5 5 --output0_shape 128 96 83 83 --stride 2 2 --padding 2 2 --dilation 1 1 --json_file=$1/${prefix}depthwise_128_96_165_165_5_5_2_SAME.json
python3 ${PARSE} --op_type DepthwiseConv2dNative --source_file $1/${prefix}depthwise_128_84_83_83_7_7_2_SAME.cc --input0_shape 128 84 83 83 --input1_shape 84 1 7 7 --output0_shape 128 84 42 42 --stride 2 2 --padding 3 3 --dilation 1 1 --json_file=$1/${prefix}depthwise_128_84_83_83_7_7_2_SAME.json
python3 ${PARSE} --op_type DepthwiseConv2dNative --source_file $1/${prefix}depthwise_128_84_42_42_7_7_1_SAME.cc --input0_shape 128 84 42 42 --input1_shape 84 1 7 7 --output0_shape 128 84 42 42 --stride 1 1 --padding 3 3 --dilation 1 1 --json_file=$1/${prefix}depthwise_128_84_42_42_7_7_1_SAME.json
python3 ${PARSE} --op_type DepthwiseConv2dNative --source_file $1/${prefix}depthwise_128_84_83_83_5_5_2_SAME.cc --input0_shape 128 84 83 83 --input1_shape 84 1 5 5 --output0_shape 128 84 42 42 --stride 2 2 --padding 2 2 --dilation 1 1 --json_file=$1/${prefix}depthwise_128_84_83_83_5_5_2_SAME.json
python3 ${PARSE} --op_type DepthwiseConv2dNative --source_file $1/${prefix}depthwise_128_84_42_42_5_5_1_SAME.cc --input0_shape 128 84 42 42 --input1_shape 84 1 5 5 --output0_shape 128 84 42 42 --stride 1 1 --padding 2 2 --dilation 1 1 --json_file=$1/${prefix}depthwise_128_84_42_42_5_5_1_SAME.json
python3 ${PARSE} --op_type DepthwiseConv2dNative --source_file $1/${prefix}depthwise_128_84_42_42_3_3_1_SAME.cc --input0_shape 128 84 42 42 --input1_shape 84 1 3 3 --output0_shape 128 84 42 42 --stride 1 1 --padding 1 1 --dilation 1 1 --json_file=$1/${prefix}depthwise_128_84_42_42_3_3_1_SAME.json
python3 ${PARSE} --op_type DepthwiseConv2dNative --source_file $1/${prefix}depthwise_128_168_42_42_3_3_1_SAME.cc --input0_shape 128 168 42 42 --input1_shape 168 1 3 3 --output0_shape 128 168 42 42 --stride 1 1 --padding 1 1 --dilation 1 1 --json_file=$1/${prefix}depthwise_128_168_42_42_3_3_1_SAME.json
python3 ${PARSE} --op_type DepthwiseConv2dNative --source_file $1/${prefix}depthwise_128_168_42_42_5_5_1_SAME.cc --input0_shape 128 168 42 42 --input1_shape 168 1 5 5 --output0_shape 128 168 42 42 --stride 1 1 --padding 2 2 --dilation 1 1 --json_file=$1/${prefix}depthwise_128_168_42_42_5_5_1_SAME.json
python3 ${PARSE} --op_type DepthwiseConv2dNative --source_file $1/${prefix}depthwise_128_336_21_21_7_7_1_SAME.cc --input0_shape 128 336 21 21 --input1_shape 336 1 7 7 --output0_shape 128 336 21 21 --stride 1 1 --padding 3 3 --dilation 1 1 --json_file=$1/${prefix}depthwise_128_336_21_21_7_7_1_SAME.json
python3 ${PARSE} --op_type DepthwiseConv2dNative --source_file $1/${prefix}depthwise_128_336_21_21_5_5_1_SAME.cc --input0_shape 128 336 21 21 --input1_shape 336 1 5 5 --output0_shape 128 336 21 21 --stride 1 1 --padding 2 2 --dilation 1 1 --json_file=$1/${prefix}depthwise_128_336_21_21_5_5_1_SAME.json
python3 ${PARSE} --op_type DepthwiseConv2dNative --source_file $1/${prefix}depthwise_128_336_21_21_3_3_1_SAME.cc --input0_shape 128 336 21 21 --input1_shape 336 1 3 3 --output0_shape 128 336 21 21 --stride 1 1 --padding 1 1 --dilation 1 1 --json_file=$1/${prefix}depthwise_128_336_21_21_3_3_1_SAME.json
python3 ${PARSE} --op_type DepthwiseConv2dNative --source_file $1/${prefix}depthwise_128_672_21_21_7_7_2_SAME.cc --input0_shape 128 672 21 21 --input1_shape 672 1 7 7 --output0_shape 128 672 11 11 --stride 2 2 --padding 3 3 --dilation 1 1 --json_file=$1/${prefix}depthwise_128_672_21_21_7_7_2_SAME.json
python3 ${PARSE} --op_type DepthwiseConv2dNative --source_file $1/${prefix}depthwise_128_672_11_11_7_7_1_SAME.cc --input0_shape 128 672 11 11 --input1_shape 672 1 7 7 --output0_shape 128 672 11 11 --stride 1 1 --padding 3 3 --dilation 1 1 --json_file=$1/${prefix}depthwise_128_672_11_11_7_7_1_SAME.json
python3 ${PARSE} --op_type DepthwiseConv2dNative --source_file $1/${prefix}depthwise_128_672_21_21_5_5_2_SAME.cc --input0_shape 128 672 21 21 --input1_shape 672 1 5 5 --output0_shape 128 672 11 11 --stride 2 2 --padding 2 2 --dilation 1 1 --json_file=$1/${prefix}depthwise_128_672_21_21_5_5_2_SAME.json
python3 ${PARSE} --op_type DepthwiseConv2dNative --source_file $1/${prefix}depthwise_128_672_11_11_5_5_1_SAME.cc --input0_shape 128 672 11 11 --input1_shape 672 1 5 5 --output0_shape 128 672 11 11 --stride 1 1 --padding 2 2 --dilation 1 1 --json_file=$1/${prefix}depthwise_128_672_11_11_5_5_1_SAME.json
python3 ${PARSE} --op_type DepthwiseConv2dNative --source_file $1/${prefix}depthwise_128_672_11_11_3_3_1_SAME.cc --input0_shape 128 672 11 11 --input1_shape 672 1 3 3 --output0_shape 128 672 11 11 --stride 1 1 --padding 1 1 --dilation 1 1 --json_file=$1/${prefix}depthwise_128_672_11_11_3_3_1_SAME.json
python3 ${PARSE} --op_type DepthwiseConv2dNative --source_file $1/${prefix}depthwise_128_336_47_47_7_7_2_VALID.cc --input0_shape 128 336 47 47 --input1_shape 336 1 7 7 --output0_shape 128 336 21 21 --stride 2 2 --padding 0 0 --dilation 2 2 --json_file=$1/${prefix}depthwise_128_336_47_47_7_7_2_VALID.json
python3 ${PARSE} --op_type DepthwiseConv2dNative --source_file $1/${prefix}depthwise_128_336_45_45_5_5_2_VALID.cc --input0_shape 128 336 45 45 --input1_shape 336 1 5 5 --output0_shape 128 336 21 21 --stride 2 2 --padding 0 0 --dilation 2 2 --json_file=$1/${prefix}depthwise_128_336_45_45_5_5_2_VALID.json
python3 ${PARSE} --op_type Dot --source_file $1/${prefix}matmul_128_4032_1000.cc --input0_shape 128 4032 --input1_shape 4032 1000 --output0_shape 128 1000 --transpose_A False --transpose_B False --json_file=$1/${prefix}matmul_128_4032_1000.json

python3 ${INJ} $1/${prefix}conv2d_128_3_331_331_96_3_3_2_VALID_add_relu.json
python3 ${INJ} $1/${prefix}conv2d_128_96_165_165_42_1_1_1_SAME_add.json
python3 ${INJ} $1/${prefix}conv2d_128_96_83_83_42_1_1_1_SAME_add_relu.json
python3 ${INJ} $1/${prefix}conv2d_128_42_83_83_42_1_1_1_VALID_add.json
python3 ${INJ} $1/${prefix}conv2d_128_168_83_83_84_1_1_1_SAME_add.json
python3 ${INJ} $1/${prefix}conv2d_128_96_83_83_42_1_1_1_SAME.json
python3 ${INJ} $1/${prefix}conv2d_128_84_42_42_84_1_1_1_VALID_add_relu.json
python3 ${INJ} $1/${prefix}conv2d_128_336_42_42_168_1_1_1_SAME_add.json
python3 ${INJ} $1/${prefix}conv2d_128_168_42_42_168_1_1_1_VALID_add_relu.json
python3 ${INJ} $1/${prefix}conv2d_128_168_42_42_84_1_1_1_SAME.json
python3 ${INJ} $1/${prefix}conv2d_128_1008_42_42_168_1_1_1_SAME_add.json
python3 ${INJ} $1/${prefix}conv2d_128_1008_42_42_336_1_1_1_SAME_add.json
python3 ${INJ} $1/${prefix}conv2d_128_336_21_21_336_1_1_1_VALID_add_relu.json
python3 ${INJ} $1/${prefix}conv2d_128_1344_21_21_336_1_1_1_SAME_add.json
python3 ${INJ} $1/${prefix}conv2d_128_1008_21_21_168_1_1_1_SAME.json
python3 ${INJ} $1/${prefix}conv2d_128_2016_21_21_336_1_1_1_SAME_add.json
python3 ${INJ} $1/${prefix}conv2d_128_2016_21_21_672_1_1_1_SAME_add.json
python3 ${INJ} $1/${prefix}conv2d_128_672_11_11_672_1_1_1_VALID_add_relu.json
python3 ${INJ} $1/${prefix}conv2d_128_2688_11_11_672_1_1_1_SAME_add.json
python3 ${INJ} $1/${prefix}conv2d_128_2016_11_11_336_1_1_1_SAME.json
python3 ${INJ} $1/${prefix}conv2d_128_4032_11_11_672_1_1_1_SAME_add.json
python3 ${INJ} $1/${prefix}depthwise_128_96_165_165_7_7_2_SAME.json
python3 ${INJ} $1/${prefix}depthwise_128_42_83_83_7_7_1_SAME.json
python3 ${INJ} $1/${prefix}depthwise_128_42_165_165_5_5_2_SAME.json
python3 ${INJ} $1/${prefix}depthwise_128_42_83_83_5_5_1_SAME.json
python3 ${INJ} $1/${prefix}depthwise_128_42_83_83_3_3_1_SAME.json
python3 ${INJ} $1/${prefix}depthwise_128_96_165_165_5_5_2_SAME.json
python3 ${INJ} $1/${prefix}depthwise_128_84_83_83_7_7_2_SAME.json
python3 ${INJ} $1/${prefix}depthwise_128_84_42_42_7_7_1_SAME.json
python3 ${INJ} $1/${prefix}depthwise_128_84_83_83_5_5_2_SAME.json
python3 ${INJ} $1/${prefix}depthwise_128_84_42_42_5_5_1_SAME.json
python3 ${INJ} $1/${prefix}depthwise_128_84_42_42_3_3_1_SAME.json
python3 ${INJ} $1/${prefix}depthwise_128_168_42_42_3_3_1_SAME.json
python3 ${INJ} $1/${prefix}depthwise_128_168_42_42_5_5_1_SAME.json
python3 ${INJ} $1/${prefix}depthwise_128_336_21_21_7_7_1_SAME.json
python3 ${INJ} $1/${prefix}depthwise_128_336_21_21_5_5_1_SAME.json
python3 ${INJ} $1/${prefix}depthwise_128_336_21_21_3_3_1_SAME.json
python3 ${INJ} $1/${prefix}depthwise_128_672_21_21_7_7_2_SAME.json
python3 ${INJ} $1/${prefix}depthwise_128_672_11_11_7_7_1_SAME.json
python3 ${INJ} $1/${prefix}depthwise_128_672_21_21_5_5_2_SAME.json
python3 ${INJ} $1/${prefix}depthwise_128_672_11_11_5_5_1_SAME.json
python3 ${INJ} $1/${prefix}depthwise_128_672_11_11_3_3_1_SAME.json
python3 ${INJ} $1/${prefix}depthwise_128_336_47_47_7_7_2_VALID.json
python3 ${INJ} $1/${prefix}depthwise_128_336_45_45_5_5_2_VALID.json
python3 ${INJ} $1/${prefix}matmul_128_4032_1000.json
