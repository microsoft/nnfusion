python3 -u $1/matmul_tunning.py 65536 2 1024 $2
python3 -u $1/matmul_tunning.py 128 4032 1000 $2
python3 -u $1/matmul_tunning.py 128 2048 1000 $2
python3 -u $1/matmul_tunning.py 65536 1024 4096 $2
python3 -u $1/matmul_tunning.py 65536 1024 1024 $2
python3 -u $1/matmul_tunning.py 65536 4096 1024 $2
python3 -u $1/matmul_tunning.py 65536 30522 1024 $2