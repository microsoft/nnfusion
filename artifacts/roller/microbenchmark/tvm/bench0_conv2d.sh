# N, CI, H, W, CO, KH, KW, strides, dilation, padding
# padding='valid' is the same as no padding. padding='same' pads the input so the output has the shape as the input. 
# However, this mode doesn’t support any stride values other than 1.

python3 $1/conv2d_tuning.py 128 128 28 28 128 3 3 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 128 58 58 128 3 3 2 1 VALID $2
python3 $1/conv2d_tuning.py 128 256 30 30 256 3 3 2 1 VALID $2
python3 $1/conv2d_tuning.py 128 168 42 42 168 1 1 1 1 VALID $2
python3 $1/conv2d_tuning.py 128 512 7 7 512 3 3 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 256 14 14 1024 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 1024 14 14 256 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 1024 14 14 512 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 1008 21 21 168 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 42 83 83 42 1 1 1 1 VALID $2
python3 $1/conv2d_tuning.py 128 4032 11 11 672 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 512 16 16 512 3 3 2 1 VALID $2
python3 $1/conv2d_tuning.py 128 96 83 83 42 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 96 165 165 42 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 168 83 83 84 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 336 21 21 336 1 1 1 1 VALID $2
python3 $1/conv2d_tuning.py 128 512 28 28 1024 1 1 2 1 VALID $2
python3 $1/conv2d_tuning.py 128 64 56 56 256 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 256 56 56 64 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 128 28 28 512 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 512 28 28 128 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 168 42 42 84 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 512 28 28 256 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 64 56 56 64 3 3 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 2016 21 21 672 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 512 7 7 2048 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 2048 7 7 512 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 84 42 42 84 1 1 1 1 VALID $2
python3 $1/conv2d_tuning.py 128 336 42 42 168 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 672 11 11 672 1 1 1 1 VALID $2
python3 $1/conv2d_tuning.py 128 1024 14 14 2048 1 1 2 1 VALID $2
python3 $1/conv2d_tuning.py 128 2016 11 11 336 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 2016 21 21 336 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 1008 42 42 336 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 64 56 56 64 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 3 230 230 64 7 7 2 1 VALID $2
python3 $1/conv2d_tuning.py 128 3 331 331 96 3 3 2 1 VALID $2
python3 $1/conv2d_tuning.py 128 256 56 56 128 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 256 14 14 256 3 3 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 2688 11 11 672 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 1008 42 42 168 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 96 83 83 42 1 1 1 1 VALID $2
python3 $1/conv2d_tuning.py 128 256 56 56 512 1 1 2 1 VALID $2
python3 $1/conv2d_tuning.py 128 1344 21 21 336 1 1 1 1 SAME $2


# python3 $1/conv2d_tuning.py 1 168 42 42 84 1 1 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 128 28 28 128 3 3 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 4032 11 11 672 1 1 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 84 42 42 84 1 1 1 1 VALID $2
# python3 $1/conv2d_tuning.py 1 512 7 7 512 3 3 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 256 14 14 1024 1 1 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 1024 14 14 256 1 1 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 1024 14 14 512 1 1 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 64 56 56 64 1 1 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 256 56 56 128 1 1 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 96 165 165 42 1 1 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 336 42 42 168 1 1 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 512 16 16 512 3 3 2 1 VALID $2
# python3 $1/conv2d_tuning.py 1 3 331 331 96 3 3 2 1 VALID $2
# python3 $1/conv2d_tuning.py 1 1008 21 21 168 1 1 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 168 83 83 84 1 1 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 336 21 21 336 1 1 1 1 VALID $2
# python3 $1/conv2d_tuning.py 1 1024 14 14 2048 1 1 2 1 VALID $2
# python3 $1/conv2d_tuning.py 1 256 56 56 512 1 1 2 1 VALID $2
# python3 $1/conv2d_tuning.py 1 96 83 83 42 1 1 1 1 VALID $2
# python3 $1/conv2d_tuning.py 1 1008 42 42 168 1 1 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 64 56 56 64 3 3 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 256 14 14 256 3 3 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 2016 21 21 672 1 1 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 2688 11 11 672 1 1 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 256 56 56 64 1 1 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 64 56 56 256 1 1 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 128 28 28 512 1 1 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 512 28 28 128 1 1 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 512 28 28 256 1 1 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 42 83 83 42 1 1 1 1 VALID $2
# python3 $1/conv2d_tuning.py 1 168 42 42 168 1 1 1 1 VALID $2
# python3 $1/conv2d_tuning.py 1 2016 21 21 336 1 1 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 512 7 7 2048 1 1 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 2048 7 7 512 1 1 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 128 58 58 128 3 3 2 1 VALID $2
# python3 $1/conv2d_tuning.py 1 1008 42 42 336 1 1 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 1344 21 21 336 1 1 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 672 11 11 672 1 1 1 1 VALID $2
# python3 $1/conv2d_tuning.py 1 3 230 230 64 7 7 2 1 VALID $2
# python3 $1/conv2d_tuning.py 1 512 28 28 1024 1 1 2 1 VALID $2
# python3 $1/conv2d_tuning.py 1 2016 11 11 336 1 1 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 96 83 83 42 1 1 1 1 SAME $2
# python3 $1/conv2d_tuning.py 1 256 30 30 256 3 3 2 1 VALID $2
