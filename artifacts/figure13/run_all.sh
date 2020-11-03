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

cd resnext_imagenet_nchw_bs1_rammerbase/cuda_codegen
./main_test > ../../../logs/resnext_imagenet_nchw_bs1.rammerbase.1000.log 2>&1
cd ../..

cd nasnet_imagenet_nchw_bs1_rammerbase/cuda_codegen
./main_test > ../../../logs/nasnet_imagenet_nchw_bs1.rammerbase.1000.log 2>&1
cd ../..

cd ..

# run rammer
cd rammer/

cd resnext_imagenet_nchw_bs1_rammer/cuda_codegen
./main_test > ../../../logs/resnext_imagenet_nchw_bs1.rammer.1000.log 2>&1
cd ../..

cd nasnet_imagenet_nchw_bs1_rammer/cuda_codegen
./main_test > ../../../logs/nasnet_imagenet_nchw_bs1.rammer.1000.log 2>&1
cd ../..

cd ..