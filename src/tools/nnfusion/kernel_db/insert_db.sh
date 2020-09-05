JSONS=../../../../artifacts/kernel_db/rammer_kernels/*.json

for f in $JSONS
do
    python convert_tvm.py $f
done