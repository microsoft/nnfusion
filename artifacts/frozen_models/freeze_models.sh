echo "freeze model pbs"

source ../.profile

pip uninstall tensorflow -y
pip uninstall tensorflow-gpu -y
pip install tensorflow-gpu==1.14.0

# rm -rf frozen_pbs
mkdir frozen_pbs

cd ../models

cd resnext_nchw
python resnext_inference.py --frozen_file ../../frozen_models/frozen_pbs/frozen_resnext_nchw_infer_bs1.pb
python resnext_inference.py --batch_size 4 --frozen_file ../../frozen_models/frozen_pbs/frozen_resnext_nchw_infer_bs4.pb
python resnext_inference.py --batch_size 16 --frozen_file ../../frozen_models/frozen_pbs/frozen_resnext_nchw_infer_bs16.pb
cd ..

cd nasnet_cifar_nchw
python nasnet_cifar_inference.py --frozen_file ../../frozen_models/frozen_pbs/frozen_nasnet_cifar_nchw_infer_bs1.pb
cd ..

cd alexnet_nchw
python alexnet_inference.py --frozen_file ../../frozen_models/frozen_pbs/frozen_alexnet_nchw_infer_bs1.pb
cd ..

cd deepspeech2
python deep_speech_inference.py --frozen_file ../../frozen_models/frozen_pbs/frozen_deepspeech2_infer_bs1.pb
cd ..

cd lstm
python lstm_inference.py --frozen_file ../../frozen_models/frozen_pbs/frozen_lstm_infer_bs1.pb
python lstm_inference.py --batch_size 4 --frozen_file ../../frozen_models/frozen_pbs/frozen_lstm_infer_bs4.pb
python lstm_inference.py --batch_size 16 --frozen_file ../../frozen_models/frozen_pbs/frozen_lstm_infer_bs16.pb
cd ..

cd seq2seq
python seq2seq_inference.py --frozen_file ../../frozen_models/frozen_pbs/frozen_seq2seq_infer_bs1.pb
cd ..

echo "run const folding"

cd ../frozen_models/frozen_pbs/
python ../tf_run_const_folding.py --file frozen_resnext_nchw_infer_bs1.pb
python ../tf_run_const_folding.py --file frozen_nasnet_cifar_nchw_infer_bs1.pb
python ../tf_run_const_folding.py --file frozen_alexnet_nchw_infer_bs1.pb
python ../tf_run_const_folding.py --file frozen_deepspeech2_infer_bs1.pb
python ../tf_run_const_folding.py --file frozen_lstm_infer_bs1.pb
python ../tf_run_const_folding.py --file frozen_seq2seq_infer_bs1.pb

python ../tf_run_const_folding.py --file frozen_lstm_infer_bs4.pb
python ../tf_run_const_folding.py --file frozen_lstm_infer_bs16.pb
python ../tf_run_const_folding.py --file frozen_resnext_nchw_infer_bs4.pb
python ../tf_run_const_folding.py --file frozen_resnext_nchw_infer_bs16.pb

cd ..