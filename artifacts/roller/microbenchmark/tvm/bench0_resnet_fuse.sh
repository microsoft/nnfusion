python3 $1/conv2d_tuning.py 128 3 230 230 64 7 7 2 1 VALID add relu
python3 $1/conv2d_tuning.py 128 64 56 56 256 1 1 1 1 SAME add
python3 $1/conv2d_tuning.py 128 64 56 56 64 1 1 1 1 SAME add relu
python3 $1/conv2d_tuning.py 128 64 56 56 64 3 3 1 1 SAME add relu
python3 $1/conv2d_tuning.py 128 256 56 56 64 1 1 1 1 SAME add relu
python3 $1/conv2d_tuning.py 128 256 56 56 512 1 1 2 1 VALID add
python3 $1/conv2d_tuning.py 128 256 56 56 128 1 1 1 1 SAME add relu
python3 $1/conv2d_tuning.py 128 128 58 58 128 3 3 2 1 VALID add relu
python3 $1/conv2d_tuning.py 128 128 28 28 512 1 1 1 1 SAME add
python3 $1/conv2d_tuning.py 128 512 28 28 128 1 1 1 1 SAME add relu
python3 $1/conv2d_tuning.py 128 128 28 28 128 3 3 1 1 SAME add relu
python3 $1/conv2d_tuning.py 128 512 28 28 1024 1 1 2 1 VALID add
python3 $1/conv2d_tuning.py 128 512 28 28 256 1 1 1 1 SAME add relu
python3 $1/conv2d_tuning.py 128 256 30 30 256 3 3 2 1 VALID add relu
python3 $1/conv2d_tuning.py 128 256 14 14 1024 1 1 1 1 SAME add
python3 $1/conv2d_tuning.py 128 1024 14 14 256 1 1 1 1 SAME add relu
python3 $1/conv2d_tuning.py 128 256 14 14 256 3 3 1 1 SAME add relu
python3 $1/conv2d_tuning.py 128 1024 14 14 2048 1 1 2 1 VALID add
python3 $1/conv2d_tuning.py 128 1024 14 14 512 1 1 1 1 SAME add relu
python3 $1/conv2d_tuning.py 128 512 16 16 512 3 3 2 1 VALID add relu
python3 $1/conv2d_tuning.py 128 512 7 7 2048 1 1 1 1 SAME add
python3 $1/conv2d_tuning.py 128 2048 7 7 512 1 1 1 1 SAME add relu
python3 $1/conv2d_tuning.py 128 512 7 7 512 3 3 1 1 SAME add relu


python3 $1/matmul_tuning.py 128 2048 1000 $2

python3 $1/pooling_tuning.py max 128 64 112 112 3 2 SAME $2