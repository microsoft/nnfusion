echo "run trt"

python tf_trt_run_frozen.py --num_iter 1000 --model ResNextImagenetNchw --model_path ../../frozen_models/frozen_pbs/frozen_resnext_imagenet_nchw_infer_bs1.const_folded.pb > ../logs/resnext_imagenet_nchw_bs1.trt.1000.log 2>&1

python tf_trt_run_frozen.py --num_iter 1000 --model NasnetImagenetNchw --model_path ../../frozen_models/frozen_pbs/frozen_nasnet_imagenet_nchw_infer_bs1.const_folded.pb > ../logs/nasnet_imagenet_nchw_bs1.trt.1000.log 2>&1
