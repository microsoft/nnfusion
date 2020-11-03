echo "run tf-gpu, and tf-xla"

cd ../../models/
cd resnext_nchw
python resnext_inference.py --num_iter 1000 --batch_size 1 > ../../figure12/logs/resnext_nchw_bs1.tf.1000.log 2>&1
python resnext_inference.py --num_iter 1000 --batch_size 1 --xla True > ../../figure12/logs/resnext_nchw_bs1.xla.1000.log 2>&1
python resnext_inference.py --num_iter 1000 --batch_size 4 > ../../figure12/logs/resnext_nchw_bs4.tf.1000.log 2>&1
python resnext_inference.py --num_iter 1000 --batch_size 4 --xla True > ../../figure12/logs/resnext_nchw_bs4.xla.1000.log 2>&1
python resnext_inference.py --num_iter 1000 --batch_size 16 > ../../figure12/logs/resnext_nchw_bs16.tf.1000.log 2>&1
python resnext_inference.py --num_iter 1000 --batch_size 16 --xla True > ../../figure12/logs/resnext_nchw_bs16.xla.1000.log 2>&1
cd ..

cd lstm
python lstm_inference.py --num_iter 1000 --batch_size 1 > ../../figure12/logs/lstm_bs1.tf.1000.log 2>&1
python lstm_inference.py --num_iter 1000 --batch_size 1 --xla True > ../../figure12/logs/lstm_bs1.xla.1000.log 2>&1
python lstm_inference.py --num_iter 1000 --batch_size 4 > ../../figure12/logs/lstm_bs4.tf.1000.log 2>&1
python lstm_inference.py --num_iter 1000 --batch_size 4 --xla True > ../../figure12/logs/lstm_bs4.xla.1000.log 2>&1
python lstm_inference.py --num_iter 1000 --batch_size 16 > ../../figure12/logs/lstm_bs16.tf.1000.log 2>&1
python lstm_inference.py --num_iter 1000 --batch_size 16 --xla True > ../../figure12/logs/lstm_bs16.xla.1000.log 2>&1
cd ..
