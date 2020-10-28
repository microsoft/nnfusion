mkdir ground_truth_results/

python tf_run_frozen.py --file frozen_pbs/frozen_resnext_nchw_infer_bs1.pb > ground_truth_results/resnext_nchw_infer_bs1.ground_truth.log 2>&1
python tf_run_frozen.py --file frozen_pbs/frozen_nasnet_cifar_nchw_infer_bs1.pb > ground_truth_results/nasnet_cifar_nchw_infer_bs1.ground_truth.log 2>&1
python tf_run_frozen.py --file frozen_pbs/frozen_alexnet_nchw_infer_bs1.pb > ground_truth_results/alexnet_nchw_infer_bs1.ground_truth.log 2>&1
python tf_run_frozen.py --file frozen_pbs/frozen_deepspeech2_infer_bs1.pb > ground_truth_results/deepspeech2_infer_bs1.ground_truth.log 2>&1
python tf_run_frozen.py --file frozen_pbs/frozen_lstm_infer_bs1.pb > ground_truth_results/lstm_infer_bs1.ground_truth.log 2>&1
python tf_run_frozen.py --file frozen_pbs/frozen_seq2seq_infer_bs1.pb > ground_truth_results/seq2seq_infer_bs1.ground_truth.log 2>&1
python tf_run_frozen.py --file frozen_pbs/frozen_resnext_imagenet_nchw_infer_bs1.pb > ground_truth_results/resnext_imagenet_nchw_infer_bs1.ground_truth.log 2>&1
python tf_run_frozen.py --file frozen_pbs/frozen_nasnet_imagenet_nchw_infer_bs1.pb > ground_truth_results/nasnet_imagenet_nchw_infer_bs1.ground_truth.log 2>&1
python tf_run_frozen.py --file frozen_pbs/frozen_resnext_nchw_infer_bs4.pb > ground_truth_results/resnext_nchw_infer_bs4.ground_truth.log 2>&1
python tf_run_frozen.py --file frozen_pbs/frozen_resnext_nchw_infer_bs16.pb > ground_truth_results/resnext_nchw_infer_bs16.ground_truth.log 2>&1
python tf_run_frozen.py --file frozen_pbs/frozen_lstm_infer_bs4.pb > ground_truth_results/lstm_infer_bs4.ground_truth.log 2>&1
python tf_run_frozen.py --file frozen_pbs/frozen_lstm_infer_bs16.pb > ground_truth_results/lstm_infer_bs16.ground_truth.log 2>&1