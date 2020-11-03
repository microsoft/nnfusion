source ../.profile

pip uninstall tensorflow -y
pip uninstall tensorflow-gpu -y
pip install ../.deps/tensorflow-trt/tensorflow-1.15.2-cp36-cp36m-linux_x86_64.whl

rm -rf logs
mkdir logs


cd tf-xla/
bash run_tf_xla.sh
cd ..

cd trt/
bash run_trt.sh
cd ..

cd tvm/
bash run_tvm.sh
cd ..

# run rammer_base
cd rammer_base/

cd resnext_nchw_bs1_rammerbase/cuda_codegen
./main_test > ../../../logs/resnext_nchw_bs1.rammerbase.1000.log 2>&1
cd ../..

cd nasnet_cifar_nchw_bs1_rammerbase/cuda_codegen
./main_test > ../../../logs/nasnet_cifar_nchw_bs1.rammerbase.1000.log 2>&1
cd ../..

cd alexnet_nchw_bs1_rammerbase/cuda_codegen
./main_test > ../../../logs/alexnet_nchw_bs1.rammerbase.1000.log 2>&1
cd ../..

cd deepspeech2_bs1_rammerbase/cuda_codegen
./main_test > ../../../logs/deepspeech2_bs1.rammerbase.1000.log 2>&1
cd ../..

cd lstm_bs1_rammerbase/cuda_codegen
./main_test > ../../../logs/lstm_bs1.rammerbase.1000.log 2>&1
cd ../..

cd seq2seq_bs1_rammerbase/cuda_codegen
./main_test > ../../../logs/seq2seq_bs1.rammerbase.1000.log 2>&1
cd ../..

cd ..

# run rammer
cd rammer/

cd resnext_nchw_bs1_rammer/cuda_codegen
./main_test > ../../../logs/resnext_nchw_bs1.rammer.1000.log 2>&1
cd ../..

cd nasnet_cifar_nchw_bs1_rammer/cuda_codegen
./main_test > ../../../logs/nasnet_cifar_nchw_bs1.rammer.1000.log 2>&1
cd ../..

cd alexnet_nchw_bs1_rammer/cuda_codegen
./main_test > ../../../logs/alexnet_nchw_bs1.rammer.1000.log 2>&1
cd ../..

cd deepspeech2_bs1_rammer/cuda_codegen
./main_test > ../../../logs/deepspeech2_bs1.rammer.1000.log 2>&1
cd ../..

cd lstm_bs1_rammer/cuda_codegen
./main_test > ../../../logs/lstm_bs1.rammer.1000.log 2>&1
cd ../..

cd seq2seq_bs1_rammer/cuda_codegen
./main_test > ../../../logs/seq2seq_bs1.rammer.1000.log 2>&1
cd ../..

cd ..