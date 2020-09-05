bash import_autotvm_kernels.sh

python tvm_run_frozen.py --num_iter 1000 --model ResNextNchw --model_path ../../frozen_models/frozen_pbs/frozen_resnext_nchw_infer_bs1.pb --autotvm_log ./autotvm_kernels/relay_resnext_nchw_tuned.1000.log > ../logs/resnext_nchw_bs1.tvm.1000.log 2>&1

python tvm_run_frozen.py --num_iter 1000 --model NasnetCifarNchw --model_path ../../frozen_models/frozen_pbs/frozen_nasnet_cifar_nchw_infer_bs1.pb --autotvm_log ./autotvm_kernels/relay_nasnet_cifar_nchw_tuned.1000.log > ../logs/nasnet_cifar_nchw_bs1.tvm.1000.log 2>&1

python tvm_run_frozen.py --num_iter 1000 --model AlexnetNchw --model_path ../../frozen_models/frozen_pbs/frozen_alexnet_nchw_infer_bs1.pb --autotvm_log ./autotvm_kernels/relay_alexnet_nchw_tuned.1000.log > ../logs/alexnet_nchw_bs1.tvm.1000.log 2>&1

python tvm_run_frozen.py --num_iter 1000 --model DeepSpeech --model_path ../../frozen_models/frozen_pbs/frozen_deepspeech2_infer_bs1.pb --autotvm_log ./autotvm_kernels/relay_deepspeech2_tuned.1000.log > ../logs/deepspeech2_bs1.tvm.1000.log 2>&1

python tvm_run_frozen.py --num_iter 1000 --model LSTM --model_path ../../frozen_models/frozen_pbs/frozen_lstm_infer_bs1.pb --autotvm_log ./autotvm_kernels/relay_lstm_tuned.1000.log > ../logs/lstm_bs1.tvm.1000.log 2>&1

python tvm_run_frozen.py --num_iter 1000 --model Seq2seq --model_path ../../frozen_models/frozen_pbs/frozen_seq2seq_infer_bs1.pb --autotvm_log ./autotvm_kernels/relay_seq2seq_tuned.1000.log > ../logs/seq2seq_bs1.tvm.1000.log 2>&1
