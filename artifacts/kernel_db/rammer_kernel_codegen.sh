source ../.profile
export TVM_HOME=$ARTIFACTS_HOME/.deps/tvm-0.7-codegen
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/nnvm/python:${PYTHONPATH}

mkdir rammer_kernels

cd autotvm_scripts/
# topi default (TVM fallback)
# conv2d
python default_topi_conv2d_codegen.py --input_path ./op_configs/resnext_conv_kernels.txt --output_path ../rammer_kernels/rexnext_nchw_conv2d_topi_default.json
python default_topi_conv2d_codegen.py --input_path ./op_configs/resnext_imagenet_conv_kernels.txt --output_path ../rammer_kernels/resnext_imagenet_nchw_conv2d_topi_default.json
# fused_conv2d_relu
python default_topi_fused_conv2d_relu_codegen.py --input_path ./op_configs/resnext_conv_kernels.txt --output_path ../rammer_kernels/rexnext_nchw_fused_conv2d_relu_topi_default.json
python default_topi_fused_conv2d_relu_codegen.py --input_path ./op_configs/resnext_imagenet_conv_kernels.txt --output_path ../rammer_kernels/resnext_imagenet_nchw_fused_conv2d_relu_topi_default.json
# fused_conv2d_add_relu
python default_topi_fused_conv2d_add_relu_codegen.py --input_path ./op_configs/resnext_conv_kernels.txt --output_path ../rammer_kernels/rexnext_nchw_fused_conv2d_add_relu_topi_default.json
python default_topi_fused_conv2d_add_relu_codegen.py --input_path ./op_configs/resnext_imagenet_conv_kernels.txt --output_path ../rammer_kernels/resnext_imagenet_nchw_fused_conv2d_add_relu_topi_default.json

# topi tune best
# conv2d
python tune_topi_conv2d_codegen.py --input_path ./op_configs/resnext_conv_kernels.txt --autotvm_log ../autotvm_logs/resnext_nchw_tuned_conv2d.1000.log --output_path ../rammer_kernels/rexnext_nchw_conv2d_topi_tune_best.json
python tune_topi_conv2d_codegen.py --input_path ./op_configs/nasnet_cifar_conv_kernels.txt --autotvm_log ../autotvm_logs/nasnet_cifar_nchw_tuned_conv2d.1000.log --output_path ../rammer_kernels/nasnet_cifar_nchw_conv2d_topi_tune_best.json
python tune_topi_conv2d_codegen.py --input_path ./op_configs/alexnet_conv_kernels.txt --autotvm_log ../autotvm_logs/alexnet_nchw_tuned_conv2d.1000.log --output_path ../rammer_kernels/alexnet_nchw_conv2d_topi_tune_best.json
python tune_topi_conv2d_codegen.py --input_path ./op_configs/deepspeech2_conv_kernels.txt --autotvm_log ../autotvm_logs/deepspeech2_tuned_conv2d.1000.log --output_path ../rammer_kernels/deepspeech2_conv2d_topi_tune_best.json
python tune_topi_conv2d_codegen.py --input_path ./op_configs/resnext_bs4_conv_kernels.txt --autotvm_log ../autotvm_logs/resnext_nchw_bs4_tuned_conv2d.1000.log --output_path ../rammer_kernels/rexnext_nchw_bs4_conv2d_topi_tune_best.json
python tune_topi_conv2d_codegen.py --input_path ./op_configs/resnext_bs16_conv_kernels.txt --autotvm_log ../autotvm_logs/resnext_nchw_bs16_tuned_conv2d.1000.log --output_path ../rammer_kernels/rexnext_nchw_bs16_conv2d_topi_tune_best.json
python tune_topi_conv2d_codegen.py --input_path ./op_configs/resnext_imagenet_conv_kernels.txt --autotvm_log ../autotvm_logs/resnext_imagenet_nchw_tuned_conv2d.1000.log --output_path ../rammer_kernels/resnext_imagenet_nchw_conv2d_topi_tune_best.json
python tune_topi_conv2d_codegen.py --input_path ./op_configs/nasnet_imagenet_conv_kernels.txt --autotvm_log ../autotvm_logs/nasnet_imagenet_nchw_tuned_conv2d.1000.log --output_path ../rammer_kernels/nasnet_imagenet_nchw_conv2d_topi_tune_best.json
# fused_conv2d_relu
python tune_topi_fused_conv2d_relu_codegen.py --input_path ./op_configs/resnext_conv_kernels.txt --autotvm_log ../autotvm_logs/resnext_nchw_tuned_conv2d.1000.log --output_path ../rammer_kernels/rexnext_nchw_fused_conv2d_relu_topi_tune_best.json
python tune_topi_fused_conv2d_relu_codegen.py --input_path ./op_configs/nasnet_cifar_conv_kernels.txt --autotvm_log ../autotvm_logs/nasnet_cifar_nchw_tuned_conv2d.1000.log --output_path ../rammer_kernels/nasnet_cifar_nchw_fused_conv2d_relu_topi_tune_best.json
python tune_topi_fused_conv2d_relu_codegen.py --input_path ./op_configs/alexnet_conv_kernels.txt --autotvm_log ../autotvm_logs/alexnet_nchw_tuned_conv2d.1000.log --output_path ../rammer_kernels/alexnet_nchw_fused_conv2d_relu_topi_tune_best.json
python tune_topi_fused_conv2d_relu_codegen.py --input_path ./op_configs/resnext_bs4_conv_kernels.txt --autotvm_log ../autotvm_logs/resnext_nchw_bs4_tuned_conv2d.1000.log --output_path ../rammer_kernels/rexnext_nchw_bs4_fused_conv2d_relu_topi_tune_best.json
python tune_topi_fused_conv2d_relu_codegen.py --input_path ./op_configs/resnext_bs16_conv_kernels.txt --autotvm_log ../autotvm_logs/resnext_nchw_bs16_tuned_conv2d.1000.log --output_path ../rammer_kernels/rexnext_nchw_bs16_fused_conv2d_relu_topi_tune_best.json
python tune_topi_fused_conv2d_relu_codegen.py --input_path ./op_configs/resnext_imagenet_conv_kernels.txt --autotvm_log ../autotvm_logs/resnext_imagenet_nchw_tuned_conv2d.1000.log --output_path ../rammer_kernels/resnext_imagenet_nchw_fused_conv2d_relu_topi_tune_best.json
python tune_topi_fused_conv2d_relu_codegen.py --input_path ./op_configs/nasnet_imagenet_conv_kernels.txt --autotvm_log ../autotvm_logs/nasnet_imagenet_nchw_tuned_conv2d.1000.log --output_path ../rammer_kernels/nasnet_imagenet_nchw_fused_conv2d_relu_topi_tune_best.json
# fused_conv2d_add_relu
python tune_topi_fused_conv2d_add_relu_codegen.py --input_path ./op_configs/resnext_conv_kernels.txt --autotvm_log ../autotvm_logs/resnext_nchw_tuned_conv2d.1000.log --output_path ../rammer_kernels/rexnext_nchw_fused_conv2d_add_relu_topi_tune_best.json
python tune_topi_fused_conv2d_add_relu_codegen.py --input_path ./op_configs/nasnet_cifar_conv_kernels.txt --autotvm_log ../autotvm_logs/nasnet_cifar_nchw_tuned_conv2d.1000.log --output_path ../rammer_kernels/nasnet_cifar_nchw_fused_conv2d_add_relu_topi_tune_best.json
python tune_topi_fused_conv2d_add_relu_codegen.py --input_path ./op_configs/alexnet_conv_kernels.txt --autotvm_log ../autotvm_logs/alexnet_nchw_tuned_conv2d.1000.log --output_path ../rammer_kernels/alexnet_nchw_fused_conv2d_add_relu_topi_tune_best.json
python tune_topi_fused_conv2d_add_relu_codegen.py --input_path ./op_configs/resnext_bs4_conv_kernels.txt --autotvm_log ../autotvm_logs/resnext_nchw_bs4_tuned_conv2d.1000.log --output_path ../rammer_kernels/rexnext_nchw_bs4_fused_conv2d_add_relu_topi_tune_best.json
python tune_topi_fused_conv2d_add_relu_codegen.py --input_path ./op_configs/resnext_bs16_conv_kernels.txt --autotvm_log ../autotvm_logs/resnext_nchw_bs16_tuned_conv2d.1000.log --output_path ../rammer_kernels/rexnext_nchw_bs16_fused_conv2d_add_relu_topi_tune_best.json
python tune_topi_fused_conv2d_add_relu_codegen.py --input_path ./op_configs/resnext_imagenet_conv_kernels.txt --autotvm_log ../autotvm_logs/resnext_imagenet_nchw_tuned_conv2d.1000.log --output_path ../rammer_kernels/resnext_imagenet_nchw_fused_conv2d_add_relu_topi_tune_best.json
python tune_topi_fused_conv2d_add_relu_codegen.py --input_path ./op_configs/nasnet_imagenet_conv_kernels.txt --autotvm_log ../autotvm_logs/nasnet_imagenet_nchw_tuned_conv2d.1000.log --output_path ../rammer_kernels/nasnet_imagenet_nchw_fused_conv2d_add_relu_topi_tune_best.json
# dot
python tune_topi_dense_codegen.py --autotvm_log ../autotvm_logs/all_tuned_topi_dense.1000.log --output_path ../rammer_kernels/all_dense_nt_topi_tune_best.json
python tune_tilling_dense_codegen.py --autotvm_log ../autotvm_logs/all_tuned_tilling_dense_nn.1000.log --output_path ../rammer_kernels/all_dense_nn_tilling_tune_best.json


# efficient-kernel pre-selection for interplay
# deepspeech2, lstm, seq2seq
python tune_tilling_dense_select_codegen.py --output_path ../rammer_kernels/all_dense_nn_tilling_tune_efficient.json
# resnext
python tune_topi_conv2d_select.py --input_path ./op_configs/resnext_conv_kernels.txt --autotvm_log ../autotvm_logs/resnext_nchw_tuned_conv2d.1000.log --output_path ./tune_topi_conv2d_select.log
python tune_topi_conv2d_codegen.py --input_path ./op_configs/resnext_conv_kernels.txt --autotvm_log tune_topi_conv2d_select.log --output_path ../rammer_kernels/rexnext_nchw_conv2d_topi_tune_efficient.json
python tune_topi_fused_conv2d_relu_codegen.py --input_path ./op_configs/resnext_conv_kernels.txt --autotvm_log tune_topi_conv2d_select.log  --output_path ../rammer_kernels/rexnext_nchw_fused_conv2d_relu_topi_tune_efficient.json
python tune_topi_fused_conv2d_add_relu_codegen.py --input_path ./op_configs/resnext_conv_kernels.txt --autotvm_log tune_topi_conv2d_select.log  --output_path ../rammer_kernels/rexnext_nchw_fused_conv2d_add_relu_topi_tune_efficient.json
# nasnet-cifar
python tune_topi_conv2d_select.py --input_path ./op_configs/nasnet_cifar_conv_kernels.txt --autotvm_log ../autotvm_logs/nasnet_cifar_nchw_tuned_conv2d.1000.log --output_path ./tune_topi_conv2d_select.log
python tune_topi_conv2d_codegen.py --input_path ./op_configs/nasnet_cifar_conv_kernels.txt --autotvm_log ./tune_topi_conv2d_select.log --output_path ../rammer_kernels/nasnet_cifar_nchw_conv2d_topi_tune_efficient.json
python tune_topi_fused_conv2d_relu_codegen.py --input_path ./op_configs/nasnet_cifar_conv_kernels.txt --autotvm_log ./tune_topi_conv2d_select.log --output_path ../rammer_kernels/nasnet_cifar_nchw_fused_conv2d_relu_topi_tune_efficient.json
python tune_topi_fused_conv2d_add_relu_codegen.py --input_path ./op_configs/nasnet_cifar_conv_kernels.txt --autotvm_log ./tune_topi_conv2d_select.log --output_path ../rammer_kernels/nasnet_cifar_nchw_fused_conv2d_add_relu_topi_tune_efficient.json
# resnext_bs4
python tune_topi_conv2d_select.py --input_path ./op_configs/resnext_bs4_conv_kernels.txt --autotvm_log ../autotvm_logs/resnext_nchw_bs4_tuned_conv2d.1000.log --output_path ./tune_topi_conv2d_select.log
python tune_topi_conv2d_codegen.py --input_path ./op_configs/resnext_bs4_conv_kernels.txt --autotvm_log tune_topi_conv2d_select.log --output_path ../rammer_kernels/rexnext_nchw_bs4_conv2d_topi_tune_efficient.json
python tune_topi_fused_conv2d_relu_codegen.py --input_path ./op_configs/resnext_bs4_conv_kernels.txt --autotvm_log tune_topi_conv2d_select.log  --output_path ../rammer_kernels/rexnext_nchw_bs4_fused_conv2d_relu_topi_tune_efficient.json
python tune_topi_fused_conv2d_add_relu_codegen.py --input_path ./op_configs/resnext_bs4_conv_kernels.txt --autotvm_log tune_topi_conv2d_select.log  --output_path ../rammer_kernels/rexnext_nchw_bs4_fused_conv2d_add_relu_topi_tune_efficient.json
# resnext_bs16
python tune_topi_conv2d_select.py --input_path ./op_configs/resnext_bs16_conv_kernels.txt --autotvm_log ../autotvm_logs/resnext_nchw_bs16_tuned_conv2d.1000.log --output_path ./tune_topi_conv2d_select.log
python tune_topi_conv2d_codegen.py --input_path ./op_configs/resnext_bs16_conv_kernels.txt --autotvm_log tune_topi_conv2d_select.log --output_path ../rammer_kernels/rexnext_nchw_bs16_conv2d_topi_tune_efficient.json
python tune_topi_fused_conv2d_relu_codegen.py --input_path ./op_configs/resnext_bs16_conv_kernels.txt --autotvm_log tune_topi_conv2d_select.log  --output_path ../rammer_kernels/rexnext_nchw_bs16_fused_conv2d_relu_topi_tune_efficient.json
python tune_topi_fused_conv2d_add_relu_codegen.py --input_path ./op_configs/resnext_bs16_conv_kernels.txt --autotvm_log tune_topi_conv2d_select.log  --output_path ../rammer_kernels/rexnext_nchw_bs16_fused_conv2d_add_relu_topi_tune_efficient.json
# resnext-imagenet
python tune_topi_conv2d_select.py --input_path ./op_configs/resnext_imagenet_conv_kernels.txt --autotvm_log ../autotvm_logs/resnext_imagenet_nchw_tuned_conv2d.1000.log --output_path ./tune_topi_conv2d_select.log
python tune_topi_conv2d_codegen.py --input_path ./op_configs/resnext_imagenet_conv_kernels.txt --autotvm_log tune_topi_conv2d_select.log --output_path ../rammer_kernels/resnext_imagenet_nchw_conv2d_topi_tune_efficient.json
python tune_topi_fused_conv2d_relu_codegen.py --input_path ./op_configs/resnext_imagenet_conv_kernels.txt --autotvm_log tune_topi_conv2d_select.log  --output_path ../rammer_kernels/resnext_imagenet_nchw_fused_conv2d_relu_topi_tune_efficient.json
python tune_topi_fused_conv2d_add_relu_codegen.py --input_path ./op_configs/resnext_imagenet_conv_kernels.txt --autotvm_log tune_topi_conv2d_select.log  --output_path ../rammer_kernels/resnext_imagenet_nchw_fused_conv2d_add_relu_topi_tune_efficient.json
# nasnet-imagenet
python tune_topi_conv2d_select.py --input_path ./op_configs/nasnet_imagenet_conv_kernels.txt --autotvm_log ../autotvm_logs/nasnet_imagenet_nchw_tuned_conv2d.1000.log --output_path ./tune_topi_conv2d_select.log
python tune_topi_conv2d_codegen.py --input_path ./op_configs/nasnet_imagenet_conv_kernels.txt --autotvm_log ./tune_topi_conv2d_select.log --output_path ../rammer_kernels/nasnet_imagenet_nchw_conv2d_topi_tune_efficient.json
python tune_topi_fused_conv2d_relu_codegen.py --input_path ./op_configs/nasnet_imagenet_conv_kernels.txt --autotvm_log ./tune_topi_conv2d_select.log --output_path ../rammer_kernels/nasnet_imagenet_nchw_fused_conv2d_relu_topi_tune_efficient.json
python tune_topi_fused_conv2d_add_relu_codegen.py --input_path ./op_configs/nasnet_imagenet_conv_kernels.txt --autotvm_log ./tune_topi_conv2d_select.log --output_path ../rammer_kernels/nasnet_imagenet_nchw_fused_conv2d_add_relu_topi_tune_efficient.json
# manual kernels
cd ../manual_kernels/
python manual_dense_codegen.py --output_path ../rammer_kernels/all_dense_nn_manual.json


cd ../

source ../.profile