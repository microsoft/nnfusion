
cd tf-xla/
cd ../models/
cd bert
python bert_large_inference.py --batch_size 128 --num_iter 1000 --xla True > ../../logs/bert_large_infer_bs128.xla.1000.log 2>&1
cd ..
cd nasnet_large_nchw
python nasnet_large_inference.py --batch_size 128 --num_iter 1000 --xla True > ../../logs/nasnet_large_nchw_infer_bs128.xla.1000.log 2>&1
cd ..

cd ../trt
python tf_trt_run_frozen.py --batch_size 128 --num_iter 1000 --model BertLarge --model_path ../frozen_models/frozen_pbs/frozen_bert_large_infer_bs128.const_folded.pb > ../logs/bert_large_infer_bs128.trt.1000.log 2>&1
