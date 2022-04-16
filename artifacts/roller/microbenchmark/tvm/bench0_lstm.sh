python3 -u $1/matmul_tuning.py 128 256 256 $2

python3 -u $1/matmul_tuning.py 512 1024 4096 $2
python3 -u $1/matmul_tuning.py 1024 1024 4096 $2
python3 -u $1/matmul_tuning.py 2048 1024 4096 $2
python3 -u $1/matmul_tuning.py 4096 1024 4096 $2
python3 -u $1/matmul_tuning.py 8192 1024 4096 $2
python3 -u $1/matmul_tuning.py 16384 1024 4096 $2

