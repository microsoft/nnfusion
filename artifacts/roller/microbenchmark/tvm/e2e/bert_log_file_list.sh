if [ $1 == "ansor" ]; then
  prefix="ansor_"
else
  prefix=""
fi

cp ../$1/matmul/${prefix}matmul_65536_30522_1024.log $1
cp ../$1/matmul/${prefix}matmul_65536_2_1024.log $1
cp ../$1/matmul/${prefix}matmul_65536_1024_1024.log $1
cp ../$1/matmul/${prefix}matmul_65536_1024_4096.log $1
cp ../$1/matmul/${prefix}matmul_65536_4096_1024.log $1
cp ../$1/batch_matmul/${prefix}batch_matmul_128_16_512_512_64.log $1
cp ../$1/batch_matmul/${prefix}batch_matmul_128_16_512_64_512.log $1
