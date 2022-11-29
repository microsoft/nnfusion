mkdir -p ground_truth/
python test/python/run_onnx_test.py -f test/python/default_operators.txt -o default.csv
python test/python/run_onnx_test.py -f test/python/default_operators.txt -i -o input_as_constant.csv
python test/python/run_onnx_test.py -f test/python/default_operators.txt -i -a -o float16.csv
python test/python/run_onnx_test.py -f test/python/default_operators.txt -i -d -o float64.csv
