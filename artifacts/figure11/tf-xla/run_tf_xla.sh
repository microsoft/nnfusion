echo "run tf-gpu, and tf-xla"

cd ../../models/
cd resnext_nchw
python resnext_inference.py --num_iter 1000 > ../../figure11/logs/resnext_nchw_bs1.tf.1000.log 2>&1
python resnext_inference.py --num_iter 1000 --xla True > ../../figure11/logs/resnext_nchw_bs1.xla.1000.log 2>&1
cd ..

cd nasnet_cifar_nchw
python nasnet_cifar_inference.py --num_iter 1000 > ../../figure11/logs/nasnet_cifar_nchw_bs1.tf.1000.log 2>&1
python nasnet_cifar_inference.py --num_iter 1000 --xla True > ../../figure11/logs/nasnet_cifar_nchw_bs1.xla.1000.log 2>&1
cd ..

cd alexnet_nchw
python alexnet_inference.py --num_iter 1000 > ../../figure11/logs/alexnet_nchw_bs1.tf.1000.log 2>&1
python alexnet_inference.py --num_iter 1000 --xla True > ../../figure11/logs/alexnet_nchw_bs1.xla.1000.log 2>&1
cd ..

cd deepspeech2
python deep_speech_inference.py --num_iter 1000 > ../../figure11/logs/deepspeech2_bs1.tf.1000.log 2>&1
python deep_speech_inference.py --num_iter 1000 --xla True > ../../figure11/logs/deepspeech2_bs1.xla.1000.log 2>&1
cd ..

cd lstm
python lstm_inference.py --num_iter 1000 > ../../figure11/logs/lstm_bs1.tf.1000.log 2>&1
python lstm_inference.py --num_iter 1000 --xla True > ../../figure11/logs/lstm_bs1.xla.1000.log 2>&1
cd ..

cd seq2seq
python seq2seq_inference.py --num_iter 1000 > ../../figure11/logs/seq2seq_bs1.tf.1000.log 2>&1
python seq2seq_inference.py --num_iter 1000 --xla True > ../../figure11/logs/seq2seq_bs1.xla.1000.log 2>&1
cd ..
