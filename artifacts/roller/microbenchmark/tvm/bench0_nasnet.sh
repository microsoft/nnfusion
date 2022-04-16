# already tune just codegen
python3 $1/conv2d_tuning.py 128 3 331 331 96 3 3 2 1 VALID $2
python3 $1/conv2d_tuning.py 128 96 165 165 42 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 96 83 83 42 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 42 83 83 42 1 1 1 1 VALID $2
python3 $1/conv2d_tuning.py 128 168 83 83 84 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 96 83 83 42 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 84 42 42 84 1 1 1 1 VALID $2
python3 $1/conv2d_tuning.py 128 336 42 42 168 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 168 42 42 168 1 1 1 1 VALID $2
python3 $1/conv2d_tuning.py 128 168 42 42 84 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 1008 42 42 168 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 1008 42 42 336 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 336 21 21 336 1 1 1 1 VALID $2
python3 $1/conv2d_tuning.py 128 1344 21 21 336 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 1008 21 21 168 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 2016 21 21 336 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 2016 21 21 672 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 672 11 11 672 1 1 1 1 VALID $2
python3 $1/conv2d_tuning.py 128 2688 11 11 672 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 2016 11 11 336 1 1 1 1 SAME $2
python3 $1/conv2d_tuning.py 128 4032 11 11 672 1 1 1 1 SAME $2








python3 $1/depthwise_tuning.py 128 96 165 165 7 7 2 1 SAME $2
python3 $1/depthwise_tuning.py 128 42 83 83 7 7 1 1 SAME $2
python3 $1/depthwise_tuning.py 128 42 165 165 5 5 2 1 SAME $2
python3 $1/depthwise_tuning.py 128 42 83 83 5 5 1 1 SAME $2
python3 $1/depthwise_tuning.py 128 42 83 83 3 3 1 1 SAME $2
python3 $1/depthwise_tuning.py 128 96 165 165 5 5 2 1 SAME $2
python3 $1/depthwise_tuning.py 128 84 83 83 7 7 2 1 SAME $2
python3 $1/depthwise_tuning.py 128 84 42 42 7 7 1 1 SAME $2
python3 $1/depthwise_tuning.py 128 84 83 83 5 5 2 1 SAME $2
python3 $1/depthwise_tuning.py 128 84 42 42 5 5 1 1 SAME $2
python3 $1/depthwise_tuning.py 128 84 42 42 3 3 1 1 SAME $2
python3 $1/depthwise_tuning.py 128 168 42 42 3 3 1 1 SAME $2
python3 $1/depthwise_tuning.py 128 168 42 42 5 5 1 1 SAME $2
python3 $1/depthwise_tuning.py 128 336 21 21 7 7 1 1 SAME $2
python3 $1/depthwise_tuning.py 128 336 21 21 5 5 1 1 SAME $2
python3 $1/depthwise_tuning.py 128 336 21 21 3 3 1 1 SAME $2
python3 $1/depthwise_tuning.py 128 672 21 21 7 7 2 1 SAME $2
python3 $1/depthwise_tuning.py 128 672 11 11 7 7 1 1 SAME $2
python3 $1/depthwise_tuning.py 128 672 21 21 5 5 2 1 SAME $2
python3 $1/depthwise_tuning.py 128 672 11 11 5 5 1 1 SAME $2
python3 $1/depthwise_tuning.py 128 672 11 11 3 3 1 1 SAME $2

python3 $1/depthwise_tuning.py 128 336 47 47 7 7 2 2 VALID $2
python3 $1/depthwise_tuning.py 128 336 45 45 5 5 2 2 VALID $2

python3 $1/matmul_tuning.py 128 4032 1000 $2

python3 $1/pooling_tuning.py avg 128 168 83 83 1 2 VALID $2
python3 $1/pooling_tuning.py avg 128 672 21 21 3 2 SAME $2
python3 $1/pooling_tuning.py avg 128 42 83 83 3 1 SAME $2
python3 $1/pooling_tuning.py avg 128 1008 42 42 1 2 VALID $2
python3 $1/pooling_tuning.py avg 128 336 42 42 3 2 SAME $2
python3 $1/pooling_tuning.py avg 128 84 83 83 3 2 SAME $2
python3 $1/pooling_tuning.py avg 128 672 11 11 3 1 SAME $2
python3 $1/pooling_tuning.py avg 128 96 165 165 1 2 VALID $2
python3 $1/pooling_tuning.py avg 128 2016 21 21 1 2 VALID $2
python3 $1/pooling_tuning.py avg 128 42 165 165 3 2 SAME $2
python3 $1/pooling_tuning.py avg 128 84 42 42 3 1 SAME $2
python3 $1/pooling_tuning.py avg 128 336 21 21 3 1 SAME $2
python3 $1/pooling_tuning.py avg 128 168 42 42 3 1 SAME $2
