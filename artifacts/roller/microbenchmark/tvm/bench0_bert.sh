# python3 $1/matmul_tuning.py 65536 30522 1024 $2
# python3 $1/matmul_tuning.py 65536 2 1024 $2
# python3 $1/matmul_tuning.py 65536 1024 1024 $2 
# python3 $1/matmul_tuning.py 65536 1024 4096 $2 
# python3 $1/matmul_tuning.py 65536 4096 1024 $2

python3 $1/batch_matmul_tuning.py 128 16 512 512 64 $2
python3 $1/batch_matmul_tuning.py 128 16 512 64 512 $2
