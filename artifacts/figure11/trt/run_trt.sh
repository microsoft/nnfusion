echo "run trt"

python tf_trt_run_frozen.py --num_iter 1000 --model ResNextNchw --model_path ../../frozen_models/frozen_pbs/frozen_resnext_nchw_infer_bs1.const_folded.pb > ../logs/resnext_nchw_bs1.trt.1000.log 2>&1

python tf_trt_run_frozen.py --num_iter 1000 --model NasnetCifarNchw --model_path ../../frozen_models/frozen_pbs/frozen_nasnet_cifar_nchw_infer_bs1.const_folded.pb > ../logs/nasnet_cifar_nchw_bs1.trt.1000.log 2>&1

python tf_trt_run_frozen.py --num_iter 1000 --model AlexnetNchw --model_path ../../frozen_models/frozen_pbs/frozen_alexnet_nchw_infer_bs1.const_folded.pb > ../logs/alexnet_nchw_bs1.trt.1000.log 2>&1

cd deepspeech2_native/deepspeech/
make -j
bin/deepspeech > ../../../logs/deepspeech2_bs1.trt.1000.log 2>&1
cd ../..

cd lstm_native/lstm/
make -j
bin/lstm > ../../../logs/lstm_bs1.trt.1000.log 2>&1
cd ../..

cd seq2seq_native/seq2seq/
make -j
bin/seq2seq > ../../../logs/seq2seq_bs1.trt.1000.log 2>&1
cd ../..