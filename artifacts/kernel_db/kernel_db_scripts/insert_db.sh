rm ~/.cache/nnfusion/kernel_cache.db

mkdir -p ~/.cache/nnfusion/profile
cp Makefile ~/.cache/nnfusion/profile

JSONS=../rammer_kernels/*.json

for f in $JSONS
do
    python convert_tvm.py $f
done

cp ~/.cache/nnfusion/kernel_cache.db ../kernel_cache.db