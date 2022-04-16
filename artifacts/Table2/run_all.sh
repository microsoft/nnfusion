source .profile

# pip uninstall tensorflow -y
# pip uninstall tensorflow-gpu -y
# pip install ../.deps/tensorflow-trt/tensorflow-1.15.2-cp36-cp36m-linux_x86_64.whl


# Step1: Generate Roller kernels and compile end-to-end models with NNFusion(Rammer). 
# NOTE: this whole step will take long time.
source ../scripts/profile_tvm_codegen.profile
bash codegen_and_build.sh


# Step2: Run Tensorflow and TF-XLA baselines
mkdir logs

cd tf-xla/
bash run_tf_xla.sh
cd ..

# Step3: Run TensorRT baseline
cd trt/
bash run_trt.sh
cd ..

# Step4: Run TVM and Ansor baseline
source ../scripts/profile_tvm.profile
bash run_ansor_autotvm.sh

# Step5: Run Rammer+Roller code that compiled by Step1.
cd rammer_roller/

cd bert_large_infer_bs128/cuda_codegen
./main_test > ../../../logs/bert_large_infer_bs128.rammer_roller.1000.log 2>&1
cd ../..

cd nasnet_large_nchw_infer_bs128/cuda_codegen
./main_test > ../../../logs/nasnet_large_nchw_infer_bs128.rammer_roller.1000.log 2>&1
cd ../..

cd lstm_infer_bs128/cuda_codegen
./main_test > ../../../logs/lstm_infer_bs128.rammer_roller.1000.log 2>&1
cd ../..

cd resnet50_infer_bs128/cuda_codegen
./main_test > ../../../logs/resnet50_infer_bs128.rammer_roller.1000.log 2>&1
cd ../..

cd ..

# Step6: process log and print Table2
python parse_log.py
