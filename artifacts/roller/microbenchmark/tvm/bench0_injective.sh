CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 128 1008 42 42 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 128 256 14 14 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 128 1024 14 14 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 128 512 14 14 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 128 96 165 165 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 128 1344 21 21 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 128 2688 11 11 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 128 64 112 112 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 128 256 56 56 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 128 128 28 28 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 128 512 28 28 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 128 256 28 28 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 128 2016 21 21 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 128 672 11 11 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 128 168 83 83 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 128 64 56 56 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 128 168 42 42 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 128 336 21 21 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 128 4032 11 11 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 128 512 7 7 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 128 2048 7 7 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 128 84 83 83 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 128 336 42 42 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 128 42 165 165 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 128 672 21 21 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 128 84 42 42 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 128 42 83 83 $2
CUDA_VISIBLE_DEVICES=0 python3 $1/injective_tuning.py relu 128 128 56 56 $2

# python3 $1/injective_tuning.py add 128 512 1024 $2
# python3 $1/injective_tuning.py add 128 168 42 42 $2
# python3 $1/injective_tuning.py add 128 42 83 83 $2
# python3 $1/injective_tuning.py add 128 512 1024 $2
# python3 $1/injective_tuning.py add 65536 1024 $2
# python3 $1/injective_tuning.py add 128 256 56 56 $2
# python3 $1/injective_tuning.py add 128 512 28 28 $2
# python3 $1/injective_tuning.py add 128 1024 14 14 $2
# python3 $1/injective_tuning.py add 128 84 42 42 $2
# python3 $1/injective_tuning.py add 128 2048 7 7 $2
# python3 $1/injective_tuning.py add 128 336 21 21 $2
# python3 $1/injective_tuning.py add 128 672 11 11 $2
# python3 $1/injective_tuning.py add 128 16 512 512 $2

# python3 $1/injective_tuning.py sub 65536 1024 $2
# python3 $1/injective_tuning.py sub 128 512 1024 $2

# python3 $1/injective_tuning.py mul (128, 512, 1024) $2
# python3 $1/injective_tuning.py mul (65536, 1024) $2
# python3 $1/injective_tuning.py mul (65536, 4096) $2
# python3 $1/injective_tuning.py mul (128, 512, 512) $2
# python3 $1/injective_tuning.py mul (65536, 1024) $2
# python3 $1/injective_tuning.py mul (128, 512, 1024) $2
# python3 $1/injective_tuning.py mul  $2
# python3 $1/injective_tuning.py mul  $2