CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 1024 1024 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 2048 1024 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 4096 1024 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 8192 1024 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 16384 1024 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 32768 1024 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 65536 1024 $2