echo "run tf-gpu, and tf-xla"

cd ../models/

cd bert
python bert_large_inference.py --batch_size 128 --num_iter 100 > ../../logs/bert_large_infer_bs128.tf.1000.log 2>&1
cd ..

cd lstm
python lstm_inference.py --batch_size 128 --num_iter 100 > ../../logs/lstm_infer_bs128.tf.1000.log 2>&1
python lstm_inference.py --batch_size 128 --num_iter 100 --xla True > ../../logs/lstm_infer_bs128.xla.1000.log 2>&1
cd ..

cd nasnet_large_nchw
python nasnet_large_inference.py --batch_size 128 --num_iter 100 > ../../logs/nasnet_large_nchw_infer_bs128.tf.1000.log 2>&1
cd ..

cd resnet_nchw
python resnet_inference.py --batch_size 128 --num_iter 100 > ../../logs/resnet50_infer_bs128.tf.1000.log 2>&1
python resnet_inference.py --batch_size 128 --num_iter 100 --xla True > ../../logs/resnet50_infer_bs128.xla.1000.log 2>&1
cd ..
