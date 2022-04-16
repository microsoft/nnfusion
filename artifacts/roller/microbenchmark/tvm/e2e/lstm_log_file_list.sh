if [ $1 == "ansor" ]; then
  prefix="ansor_"
else
  prefix=""
fi

cp ../$1/matmul/${prefix}matmul_128_256_256.log $1
