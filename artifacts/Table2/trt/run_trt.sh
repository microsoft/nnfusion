echo "run trt"

python tf_trt_run_frozen.py --num_iter 100 --model LSTMBS128 --model_path ../frozen_models/frozen_pbs/frozen_lstm_infer_bs128.const_folded.pb > ../logs/lstm_infer_bs128.trt.1000.log 2>&1

python tf_trt_run_frozen.py --num_iter 100 --model NasnetLargeNchwBS128 --model_path ../frozen_models/frozen_pbs/frozen_nasnet_large_nchw_infer_bs128.const_folded.pb > ../logs/nasnet_large_nchw_infer_bs128.trt.1000.log 2>&1

python tf_trt_run_frozen.py --num_iter 100 --model ResNetNchwBS128 --model_path ../frozen_models/frozen_pbs/frozen_resnet50_infer_bs128.const_folded.pb > ../logs/resnet50_infer_bs128.trt.1000.log 2>&1
