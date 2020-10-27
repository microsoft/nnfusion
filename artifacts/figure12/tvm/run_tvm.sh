bash import_autotvm_kernels.sh

python tvm_run_frozen.py --num_iter 1000 --model ResNextNchw --model_path ../../frozen_models/frozen_pbs/frozen_resnext_nchw_infer_bs1.pb --autotvm_log ./autotvm_kernels/relay_resnext_nchw_tuned.1000.log > ../logs/resnext_nchw_bs1.tvm.1000.log 2>&1

python tvm_run_frozen.py --num_iter 1000 --model ResNextNchwBS4 --model_path ../../frozen_models/frozen_pbs/frozen_resnext_nchw_infer_bs4.pb --autotvm_log ./autotvm_kernels/relay_resnext_nchw_bs4_tuned.1000.log > ../logs/resnext_nchw_bs4.tvm.1000.log 2>&1

python tvm_run_frozen.py --num_iter 1000 --model ResNextNchwBS16 --model_path ../../frozen_models/frozen_pbs/frozen_resnext_nchw_infer_bs16.pb --autotvm_log ./autotvm_kernels/relay_resnext_nchw_bs16_tuned.1000.log > ../logs/resnext_nchw_bs16.tvm.1000.log 2>&1

python tvm_run_frozen.py --num_iter 1000 --model LSTM --model_path ../../frozen_models/frozen_pbs/frozen_lstm_infer_bs1.pb --autotvm_log ./autotvm_kernels/relay_lstm_tuned.1000.log > ../logs/lstm_bs1.tvm.1000.log 2>&1

python tvm_run_frozen.py --num_iter 1000 --model LSTMBS4 --model_path ../../frozen_models/frozen_pbs/frozen_lstm_infer_bs4.pb --autotvm_log ./autotvm_kernels/relay_lstm_bs4_tuned.1000.log > ../logs/lstm_bs4.tvm.1000.log 2>&1

python tvm_run_frozen.py --num_iter 1000 --model LSTMBS16 --model_path ../../frozen_models/frozen_pbs/frozen_lstm_infer_bs16.pb --autotvm_log ./autotvm_kernels/relay_lstm_bs16_tuned.1000.log > ../logs/lstm_bs16.tvm.1000.log 2>&1
