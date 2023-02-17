mkdir -p test_snapshot/
python test/python/run_onnx_test.py -f test/python/default_operators.txt -o test_snapshot/default.csv
python test/python/run_onnx_test.py -f test/python/default_operators.txt -i -o test_snapshot/input_as_constant.csv
python test/python/run_onnx_test.py -f test/python/default_operators.txt -i -a -o test_snapshot/float16.csv
python test/python/run_onnx_test.py -f test/python/default_operators.txt -i -d -o test_snapshot/float64.csv
echo "Bad cases:"
echo "FP32/Input as parameter:"
python test/python/test_snapshot_compare.py -t test_snapshot/default.csv -g test/python/ground_truth/default.csv
echo "FP32/Input as constant:"
python test/python/test_snapshot_compare.py -t test_snapshot/input_as_constant.csv -g test/python/ground_truth/input_as_constant.csv
echo "FP16/Input as constant:"
python test/python/test_snapshot_compare.py -t test_snapshot/float16.csv -g test/python/ground_truth/float16.csv
echo "FP64/Input as constant:"
python test/python/test_snapshot_compare.py -t test_snapshot/float64.csv -g test/python/ground_truth/float64.csv