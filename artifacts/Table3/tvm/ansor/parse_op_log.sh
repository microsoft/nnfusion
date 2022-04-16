LOG_DIR=./
rm ansor_op_perf.txt

python3 -u parse_op_log.py $LOG_DIR conv 128 128 28 28 128 3 3 1 1 1 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 128 58 58 128 3 3 2 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 256 30 30 256 3 3 2 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 168 42 42 168 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 512 7 7 512   3 3 1 1 1 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 256 14 14 1024 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 1024 14 14 256 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 1024 14 14 512 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 1008 21 21 168 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 42 83 83 42 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 4032 11 11 672 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 512 16 16 512 3 3 2 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 96 83 83 42 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 96 165 165 42 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 168 83 83 84 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 336 21 21 336 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 512 28 28 1024 1 1 2 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 64 56 56 256 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 256 56 56 64 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 128 28 28 512 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 512 28 28 128 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 168 42 42 84 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 512 28 28 256 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 64 56 56 64 3 3 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 2016 21 21 672 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 512 7 7 2048 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 2048 7 7 512 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 84 42 42 84 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 336 42 42 168 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 672 11 11 672 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 1024 14 14 2048 1 1 2 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 2016 11 11 336 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 2016 21 21 336 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 1008 42 42 336 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 64 56 56 64 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 3 230 230 64 7 7 2 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 3 331 331 96 3 3 2 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 256 56 56 128 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 256 14 14 256 3 3 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 2688 11 11 672 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 1008 42 42 168 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 96 83 83 42 1 1 1 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 256 56 56 512 1 1 2 1 0 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR conv 128 1344 21 21 336 1 1 1 1 0 >> ansor_op_perf.txt 2>&1

python3 -u parse_op_log.py $LOG_DIR depthwise 128 84 83 83 5 5 2 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR depthwise 128 42 83 83 5 5 1 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR depthwise 128 336 21 21 5 5 1 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR depthwise 128 42 165 165 5 5 2 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR depthwise 128 84 83 83 7 7 2 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR depthwise 128 672 11 11 3 3 1 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR depthwise 128 168 42 42 5 5 1 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR depthwise 128 672 21 21 5 5 2 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR depthwise 128 336 21 21 3 3 1 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR depthwise 128 672 21 21 7 7 2 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR depthwise 128 42 83 83 7 7 1 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR depthwise 128 84 42 42 7 7 1 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR depthwise 128 84 42 42 5 5 1 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR depthwise 128 168 42 42 3 3 1 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR depthwise 128 672 11 11 7 7 1 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR depthwise 128 336 42 42 5 5 2 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR depthwise 128 96 165 165 5 5 2 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR depthwise 128 336 21 21 7 7 1 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR depthwise 128 336 42 42 7 7 2 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR depthwise 128 42 83 83 3 3 1 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR depthwise 128 96 165 165 7 7 2 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR depthwise 128 84 42 42 3 3 1 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR depthwise 128 672 11 11 5 5 1 SAME >> ansor_op_perf.txt 2>&1

python3 -u parse_op_log.py $LOG_DIR elementwise relu 128 1008 42 42 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR elementwise relu 128 256 14 14 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR elementwise relu 128 1024 14 14 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR elementwise relu 128 512 14 14 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR elementwise relu 128 96 165 165 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR elementwise relu 128 1344 21 21 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR elementwise relu 128 2688 11 11 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR elementwise relu 128 64 112 112 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR elementwise relu 128 256 56 56 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR elementwise relu 128 128 28 28 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR elementwise relu 128 512 28 28 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR elementwise relu 128 256 28 28 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR elementwise relu 128 2016 21 21 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR elementwise relu 128 672 11 11 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR elementwise relu 128 168 83 83 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR elementwise relu 128 64 56 56 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR elementwise relu 128 168 42 42 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR elementwise relu 128 336 21 21 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR elementwise relu 128 4032 11 11 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR elementwise relu 128 512 7 7 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR elementwise relu 128 2048 7 7 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR elementwise relu 128 84 83 83 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR elementwise relu 128 336 42 42 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR elementwise relu 128 42 165 165 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR elementwise relu 128 672 21 21 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR elementwise relu 128 84 42 42 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR elementwise relu 128 42 83 83 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR elementwise relu 128 128 56 56 >> ansor_op_perf.txt 2>&1

python3 -u parse_op_log.py $LOG_DIR pooling avg 128 168 83 83 1 2 VALID >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR pooling avg 128 672 21 21 3 2 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR pooling avg 128 42 83 83 3 1 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR pooling avg 128 1008 42 42 1 2 VALID >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR pooling avg 128 336 42 42 3 2 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR pooling avg 128 84 83 83 3 2 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR pooling avg 128 672 11 11 3 1 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR pooling avg 128 96 165 165 1 2 VALID >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR pooling avg 128 2016 21 21 1 2 VALID >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR pooling avg 128 42 165 165 3 2 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR pooling avg 128 84 42 42 3 1 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR pooling avg 128 336 21 21 3 1 SAME >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR pooling avg 128 168 42 42 3 1 SAME >> ansor_op_perf.txt 2>&1

python3 -u parse_op_log.py $LOG_DIR reduction 128 512 1024 2 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR reduction 65536 1024 1 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR reduction 128 4032 11 11 2 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR reduction 128 2048 7 7 2 >> ansor_op_perf.txt 2>&1

python3 -u parse_op_log.py $LOG_DIR matmul 65536 2 1024 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR matmul 128 4032 1000 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR matmul 128 2048 1000 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR matmul 65536 1024 4096 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR matmul 65536 1024 1024 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR matmul 65536 4096 1024 >> ansor_op_perf.txt 2>&1
python3 -u parse_op_log.py $LOG_DIR matmul 65536 30522 1024 >> ansor_op_perf.txt 2>&1