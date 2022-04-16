if [ $1 == "ansor" ]; then
  prefix="ansor_"
else
  prefix=""
fi

cp ../$1/conv/${prefix}conv2d_128_3_230_230_64_7_7_2_VALID.log $1
cp ../$1/conv/${prefix}conv2d_128_64_56_56_256_1_1_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_64_56_56_64_1_1_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_64_56_56_64_3_3_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_256_56_56_64_1_1_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_256_56_56_512_1_1_2_VALID.log $1
cp ../$1/conv/${prefix}conv2d_128_256_56_56_128_1_1_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_128_58_58_128_3_3_2_VALID.log $1
cp ../$1/conv/${prefix}conv2d_128_128_28_28_512_1_1_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_512_28_28_128_1_1_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_128_28_28_128_3_3_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_512_28_28_1024_1_1_2_VALID.log $1
cp ../$1/conv/${prefix}conv2d_128_512_28_28_256_1_1_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_256_30_30_256_3_3_2_VALID.log $1
cp ../$1/conv/${prefix}conv2d_128_256_14_14_1024_1_1_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_1024_14_14_256_1_1_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_256_14_14_256_3_3_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_1024_14_14_2048_1_1_2_VALID.log $1
cp ../$1/conv/${prefix}conv2d_128_1024_14_14_512_1_1_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_512_16_16_512_3_3_2_VALID.log $1
cp ../$1/conv/${prefix}conv2d_128_512_7_7_2048_1_1_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_2048_7_7_512_1_1_1_SAME.log $1
cp ../$1/conv/${prefix}conv2d_128_512_7_7_512_3_3_1_SAME.log $1

cp ../$1/matmul/${prefix}matmul_128_2048_1000.log $1
cp ../$1/pooling/${prefix}max_pooling_128_64_112_112_3_3_2_SAME.log $1