bash import_autotvm_kernels.sh

python tvm_run_frozen.py --num_iter 1000 --model ResNextImagenetNchw --model_path ../../frozen_models/frozen_pbs/frozen_resnext_imagenet_nchw_infer_bs1.pb --autotvm_log ./autotvm_kernels/relay_resnext_imagenet_nchw_tuned.1000.log > ../logs/resnext_imagenet_nchw_bs1.tvm.1000.log 2>&1

python tvm_run_frozen.py --num_iter 1000 --model NasnetImagenetNchw --model_path ../../frozen_models/frozen_pbs/frozen_nasnet_imagenet_nchw_infer_bs1.pb --autotvm_log ./autotvm_kernels/relay_nasnet_imagenet_nchw_tuned.1000.log > ../logs/nasnet_imagenet_nchw_bs1.tvm.1000.log 2>&1
