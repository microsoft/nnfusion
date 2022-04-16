echo "freeze model pbs"

source ../.profile

pip uninstall tensorflow -y
pip uninstall tensorflow-gpu -y
pip install tensorflow-gpu==1.14.0

rm -rf frozen_pbs
mkdir -p frozen_pbs

cd ../models

cd bert
python bert_large_inference.py --batch_size 128 --frozen_file ../../frozen_models/frozen_pbs/frozen_bert_large_infer_bs128.pb 
cd ..

cd lstm
python lstm_inference.py --batch_size 128 --frozen_file ../../frozen_models/frozen_pbs/frozen_lstm_infer_bs128.pb
cd ..

cd nasnet_large_nchw
python nasnet_large_inference.py --batch_size 128 --frozen_file ../../frozen_models/frozen_pbs/frozen_nasnet_large_nchw_infer_bs128.pb
cd ..

cd resnet_nchw
python resnet_inference.py --batch_size 128 --frozen_file ../../frozen_models/frozen_pbs/frozen_resnet50_infer_bs128.pb
cd ..

echo "run const folding"

cd ../frozen_models/frozen_pbs/
python ../tf_run_const_folding.py --file frozen_bert_large_infer_bs128.pb
python ../tf_run_const_folding.py --file frozen_lstm_infer_bs128.pb
python ../tf_run_const_folding.py --file frozen_nasnet_large_nchw_infer_bs128.pb
python ../tf_run_const_folding.py --file frozen_resnet50_infer_bs128.pb

cd ..

pip uninstall tensorflow -y
pip uninstall tensorflow-gpu -y
pip install ../../wheel/tensorflow-1.15.2-cp36-cp36m-linux_x86_64.whl