echo "run trt"

python tf_trt_run_frozen.py --num_iter 1000 --batch_size 1 --model ResNextNchw --model_path ../../frozen_models/frozen_pbs/frozen_resnext_nchw_infer_bs1.const_folded.pb > ../logs/resnext_nchw_bs1.trt.1000.log 2>&1

python tf_trt_run_frozen.py --num_iter 1000 --batch_size 4 --model ResNextNchwBS4 --model_path ../../frozen_models/frozen_pbs/frozen_resnext_nchw_infer_bs4.const_folded.pb > ../logs/resnext_nchw_bs4.trt.1000.log 2>&1

python tf_trt_run_frozen.py --num_iter 1000 --batch_size 16 --model ResNextNchwBS16 --model_path ../../frozen_models/frozen_pbs/frozen_resnext_nchw_infer_bs16.const_folded.pb > ../logs/resnext_nchw_bs16.trt.1000.log 2>&1

cd lstm_native/lstm/
make -j
bin/lstm > ../../../logs/lstm_bs1.trt.1000.log 2>&1
cd ../..

cd lstm_bs4_native/lstm/
make -j
bin/lstm > ../../../logs/lstm_bs4.trt.1000.log 2>&1
cd ../..

cd lstm_bs16_native/lstm/
make -j
bin/lstm > ../../../logs/lstm_bs16.trt.1000.log 2>&1
cd ../..
