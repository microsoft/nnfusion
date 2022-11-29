mkdir -p test_snapshot/
python test/python/run_onnx_test.py -f test/python/default_operators.txt -o test_snapshot/default.csv
python test/python/run_onnx_test.py -f test/python/default_operators.txt -i -o test_snapshot/input_as_constant.csv
python test/python/run_onnx_test.py -f test/python/default_operators.txt -i -a -o test_snapshot/float16.csv
python test/python/run_onnx_test.py -f test/python/default_operators.txt -i -d -o test_snapshot/float64.csv
