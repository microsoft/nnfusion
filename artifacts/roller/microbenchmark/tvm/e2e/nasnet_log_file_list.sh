if [ $1 == "ansor" ]; then
  prefix="ansor_"
else
  prefix=""
fi

cp ../$1/conv/${prefix}conv2d_128_3_331_331_96_3_3_2_VALID.log $1
cp ../$1/conv/${prefix}conv2d_128_96_165_165_42_1_1_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_96_83_83_42_1_1_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_42_83_83_42_1_1_1_VALID.log $1
cp ../$1/conv/${prefix}conv2d_128_168_83_83_84_1_1_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_96_83_83_42_1_1_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_84_42_42_84_1_1_1_VALID.log $1
cp ../$1/conv/${prefix}conv2d_128_336_42_42_168_1_1_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_168_42_42_168_1_1_1_VALID.log $1
cp ../$1/conv/${prefix}conv2d_128_168_42_42_84_1_1_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_1008_42_42_168_1_1_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_1008_42_42_336_1_1_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_336_21_21_336_1_1_1_VALID.log $1
cp ../$1/conv/${prefix}conv2d_128_1344_21_21_336_1_1_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_1008_21_21_168_1_1_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_2016_21_21_336_1_1_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_2016_21_21_672_1_1_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_672_11_11_672_1_1_1_VALID.log $1
cp ../$1/conv/${prefix}conv2d_128_2688_11_11_672_1_1_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_2016_11_11_336_1_1_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_4032_11_11_672_1_1_1_SAME.log $1


# cp ../$1/depthwise/${prefix}depthwise_128_96_165_165_7_7_2_SAME.log $1
# cp ../$1/depthwise/${prefix}depthwise_128_42_83_83_7_7_1_SAME.log $1
# cp ../$1/depthwise/${prefix}depthwise_128_42_165_165_5_5_2_SAME.log $1
# cp ../$1/depthwise/${prefix}depthwise_128_42_83_83_5_5_1_SAME.log $1
# cp ../$1/depthwise/${prefix}depthwise_128_42_83_83_3_3_1_SAME.log $1
# cp ../$1/depthwise/${prefix}depthwise_128_96_165_165_5_5_2_SAME.log $1
# cp ../$1/depthwise/${prefix}depthwise_128_84_83_83_7_7_2_SAME.log $1
# cp ../$1/depthwise/${prefix}depthwise_128_84_42_42_7_7_1_SAME.log $1
# cp ../$1/depthwise/${prefix}depthwise_128_84_83_83_5_5_2_SAME.log $1
# cp ../$1/depthwise/${prefix}depthwise_128_84_42_42_5_5_1_SAME.log $1
# cp ../$1/depthwise/${prefix}depthwise_128_84_42_42_3_3_1_SAME.log $1
# cp ../$1/depthwise/${prefix}depthwise_128_168_42_42_3_3_1_SAME.log $1
# cp ../$1/depthwise/${prefix}depthwise_128_168_42_42_5_5_1_SAME.log $1
# cp ../$1/depthwise/${prefix}depthwise_128_336_21_21_7_7_1_SAME.log $1
# cp ../$1/depthwise/${prefix}depthwise_128_336_21_21_5_5_1_SAME.log $1
# cp ../$1/depthwise/${prefix}depthwise_128_336_21_21_3_3_1_SAME.log $1
# cp ../$1/depthwise/${prefix}depthwise_128_672_21_21_7_7_2_SAME.log $1
# cp ../$1/depthwise/${prefix}depthwise_128_672_11_11_7_7_1_SAME.log $1
# cp ../$1/depthwise/${prefix}depthwise_128_672_21_21_5_5_2_SAME.log $1
# cp ../$1/depthwise/${prefix}depthwise_128_672_11_11_5_5_1_SAME.log $1
# cp ../$1/depthwise/${prefix}depthwise_128_672_11_11_3_3_1_SAME.log $1
# cp ../$1/depthwise/${prefix}depthwise_128_336_47_47_7_7_2_VALID.log $1
# cp ../$1/depthwise/${prefix}depthwise_128_336_45_45_5_5_2_VALID.log $1
# cp ../$1/matmul/${prefix}matmul_128_4032_1000.log $1



# cp ../$1/pooling/${prefix}avg_pooling_128_168_83_83_1_1_2_VALID.log $1
# cp ../$1/pooling/${prefix}avg_pooling_128_672_21_21_3_3_2_SAME.log $1
# cp ../$1/pooling/${prefix}avg_pooling_128_42_83_83_3_3_1_SAME.log $1
# cp ../$1/pooling/${prefix}avg_pooling_128_1008_42_42_1_1_2_VALID.log $1
# cp ../$1/pooling/${prefix}avg_pooling_128_336_42_42_3_3_2_SAME.log $1
# cp ../$1/pooling/${prefix}avg_pooling_128_84_83_83_3_3_2_SAME.log $1
# cp ../$1/pooling/${prefix}avg_pooling_128_672_11_11_3_3_1_SAME.log $1
# cp ../$1/pooling/${prefix}avg_pooling_128_96_165_165_1_1_2_VALID.log $1
# cp ../$1/pooling/${prefix}avg_pooling_128_2016_21_21_1_1_2_VALID.log $1
# cp ../$1/pooling/${prefix}avg_pooling_128_42_165_165_3_3_2_SAME.log $1
# cp ../$1/pooling/${prefix}avg_pooling_128_84_42_42_3_3_1_SAME.log $1
# cp ../$1/pooling/${prefix}avg_pooling_128_336_21_21_3_3_1_SAME.log $1
# cp ../$1/pooling/${prefix}avg_pooling_128_168_42_42_3_3_1_SAME.log $1