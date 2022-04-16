mkdir -p ground_truth_results/

python tf_run_frozen.py --file frozen_pbs/frozen_bert_large_infer_bs128.pb > ground_truth_results/frozen_bert_large_infer_bs128.ground_truth.log 2>&1
python tf_run_frozen.py --file frozen_pbs/frozen_nasnet_large_nchw_infer_bs128.pb > ground_truth_results/frozen_nasnet_large_nchw_infer_bs128.ground_truth.log 2>&1
python tf_run_frozen.py --file frozen_pbs/frozen_lstm_infer_bs128.pb > ground_truth_results/frozen_lstm_infer_bs128.ground_truth.log 2>&1
python tf_run_frozen.py --file frozen_pbs/frozen_resnet50_infer_bs128.pb > ground_truth_results/frozen_resnet50_infer_bs128.ground_truth.log 2>&1
