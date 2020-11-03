#!/bin/bash

MODEL_DIR=$ARTIFACTS_HOME/models
LOG_DIR=$ARTIFACTS_HOME/figure16/logs
STEP=1000
WARMUP=5
ITERS=$STEP+$WARMUP

rm -rf $LOG_DIR
mkdir $LOG_DIR

for BS in 1
do
    cd $MODEL_DIR/alexnet_nchw
    python alexnet_inference.py --num_iter $STEP --batch_size $BS > $LOG_DIR/alexnet_nchw_bs$BS.tf.iter_time.$ITERS.log 2>&1
    nvprof --normalized-time-unit ms --log-file $LOG_DIR/alexnet_nchw_bs$BS.tf.nvprof.$ITERS.log --csv python alexnet_inference.py --num_iter $STEP --batch_size $BS


    cd $MODEL_DIR/deepspeech2
    python deep_speech_inference.py --num_iter $STEP --batch_size $BS > $LOG_DIR/deepspeech2_bs$BS.tf.iter_time.$ITERS.log 2>&1
    nvprof --normalized-time-unit ms --profiling-semaphore-pool-size 2621440 --device-buffer-size 128 --log-file $LOG_DIR/deepspeech2_bs$BS.tf.nvprof.$ITERS.log --csv python deep_speech_inference.py --num_iter $STEP --batch_size $BS


    cd $MODEL_DIR/lstm
    python lstm_inference.py --num_iter $STEP --batch_size $BS > $LOG_DIR/lstm_bs$BS.tf.iter_time.$ITERS.log 2>&1
    nvprof --normalized-time-unit ms --log-file $LOG_DIR/lstm_bs$BS.tf.nvprof.$ITERS.log --csv python lstm_inference.py --num_iter $STEP --batch_size $BS


    cd $MODEL_DIR/nasnet_cifar_nchw
    python nasnet_cifar_inference.py --num_iter $STEP --batch_size $BS > $LOG_DIR/nasnet_cifar_nchw_bs$BS.tf.iter_time.$ITERS.log 2>&1
    nvprof --normalized-time-unit ms --log-file $LOG_DIR/nasnet_cifar_nchw_bs$BS.tf.nvprof.$ITERS.log --csv python nasnet_cifar_inference.py --num_iter $STEP --batch_size $BS


    cd $MODEL_DIR/resnext_nchw
    python resnext_inference.py --num_iter $STEP --batch_size $BS > $LOG_DIR/resnext_nchw_bs$BS.tf.iter_time.$ITERS.log 2>&1
    nvprof --normalized-time-unit ms --log-file $LOG_DIR/resnext_nchw_bs$BS.tf.nvprof.$ITERS.log --csv python resnext_inference.py --num_iter $STEP --batch_size $BS


    cd $MODEL_DIR/seq2seq
    python seq2seq_inference.py --num_iter $STEP --batch_size $BS > $LOG_DIR/seq2seq_bs$BS.tf.iter_time.$ITERS.log 2>&1
    nvprof --normalized-time-unit ms --log-file $LOG_DIR/seq2seq_bs$BS.tf.nvprof.$ITERS.log --csv python seq2seq_inference.py --num_iter $STEP --batch_size $BS
done

# rammer-base
cd $ARTIFACTS_HOME/figure16

cd trt

python tf_trt_run_frozen.py --num_iter 1000 --model ResNextNchw --model_path ../../frozen_models/frozen_pbs/frozen_resnext_nchw_infer_bs1.const_folded.pb > ../logs/resnext_nchw_bs1.trt.iter_time.$ITERS.log 2>&1
nvprof --normalized-time-unit ms --log-file $LOG_DIR/resnext_nchw_bs1.trt.nvprof.$ITERS.log --csv python tf_trt_run_frozen.py --num_iter 1000 --model ResNextNchw --model_path ../../frozen_models/frozen_pbs/frozen_resnext_nchw_infer_bs1.const_folded.pb

python tf_trt_run_frozen.py --num_iter 1000 --model NasnetCifarNchw --model_path ../../frozen_models/frozen_pbs/frozen_nasnet_cifar_nchw_infer_bs1.const_folded.pb > ../logs/nasnet_cifar_nchw_bs1.trt.iter_time.$ITERS.log 2>&1
nvprof --normalized-time-unit ms --log-file $LOG_DIR/nasnet_cifar_nchw_bs1.trt.nvprof.$ITERS.log --csv python tf_trt_run_frozen.py --num_iter 1000 --model NasnetCifarNchw --model_path ../../frozen_models/frozen_pbs/frozen_nasnet_cifar_nchw_infer_bs1.const_folded.pb

python tf_trt_run_frozen.py --num_iter 1000 --model AlexnetNchw --model_path ../../frozen_models/frozen_pbs/frozen_alexnet_nchw_infer_bs1.const_folded.pb > ../logs/alexnet_nchw_bs1.trt.iter_time.$ITERS.log 2>&1
nvprof --normalized-time-unit ms --log-file $LOG_DIR/alexnet_nchw_bs1.trt.nvprof.$ITERS.log --csv python tf_trt_run_frozen.py --num_iter 1000 --model AlexnetNchw --model_path ../../frozen_models/frozen_pbs/frozen_alexnet_nchw_infer_bs1.const_folded.pb

cd deepspeech2_native/deepspeech/
make -j
bin/deepspeech > ../../../logs/deepspeech2_bs1.trt.iter_time.$ITERS.log 2>&1
nvprof --normalized-time-unit ms --log-file $LOG_DIR/deepspeech2_bs1.trt.nvprof.$ITERS.log --csv bin/deepspeech
cd ../..

cd lstm_native/lstm/
make -j
bin/lstm > ../../../logs/lstm_bs1.trt.iter_time.$ITERS.log 2>&1
nvprof --normalized-time-unit ms --log-file $LOG_DIR/lstm_bs1.trt.nvprof.$ITERS.log --csv bin/lstm
cd ../..

cd seq2seq_native/seq2seq/
make -j
bin/seq2seq > ../../../logs/seq2seq_bs1.trt.iter_time.$ITERS.log 2>&1
nvprof --normalized-time-unit ms --log-file $LOG_DIR/seq2seq_bs1.trt.nvprof.$ITERS.log --csv bin/seq2seq
cd ../..

cd ..

cd rammer_base/resnext_nchw_bs1_rammerbase/cuda_codegen/
./main_test > $LOG_DIR/resnext_nchw_bs1.rammerbase.iter_time.$ITERS.log 2>&1
nvprof --normalized-time-unit ms --log-file $LOG_DIR/resnext_nchw_bs1.rammerbase.nvprof.$ITERS.log --csv ./main_test
cd ../../../

cd rammer_base/nasnet_cifar_nchw_bs1_rammerbase/cuda_codegen/
./main_test > $LOG_DIR/nasnet_cifar_nchw_bs1.rammerbase.iter_time.$ITERS.log 2>&1
nvprof --normalized-time-unit ms --log-file $LOG_DIR/nasnet_cifar_nchw_bs1.rammerbase.nvprof.$ITERS.log --csv ./main_test
cd ../../../

cd rammer_base/alexnet_nchw_bs1_rammerbase/cuda_codegen/
./main_test > $LOG_DIR/alexnet_nchw_bs1.rammerbase.iter_time.$ITERS.log 2>&1
nvprof --normalized-time-unit ms --log-file $LOG_DIR/alexnet_nchw_bs1.rammerbase.nvprof.$ITERS.log --csv ./main_test
cd ../../../

cd rammer_base/deepspeech2_bs1_rammerbase/cuda_codegen/
./main_test > $LOG_DIR/deepspeech2_bs1.rammerbase.iter_time.$ITERS.log 2>&1
nvprof --normalized-time-unit ms --log-file $LOG_DIR/deepspeech2_bs1.rammerbase.nvprof.$ITERS.log --csv ./main_test
cd ../../../

cd rammer_base/lstm_bs1_rammerbase/cuda_codegen/
./main_test > $LOG_DIR/lstm_bs1.rammerbase.iter_time.$ITERS.log 2>&1
nvprof --normalized-time-unit ms --log-file $LOG_DIR/lstm_bs1.rammerbase.nvprof.$ITERS.log --csv ./main_test
cd ../../../

cd rammer_base/seq2seq_bs1_rammerbase/cuda_codegen/
./main_test > $LOG_DIR/seq2seq_bs1.rammerbase.iter_time.$ITERS.log 2>&1
nvprof --normalized-time-unit ms --log-file $LOG_DIR/seq2seq_bs1.rammerbase.nvprof.$ITERS.log --csv ./main_test
cd ../../../

# run rammer

cd rammer/resnext_nchw_bs1_rammer/cuda_codegen/
./main_test > $LOG_DIR/resnext_nchw_bs1.rammer.iter_time.$ITERS.log 2>&1
nvprof --normalized-time-unit ms --log-file $LOG_DIR/resnext_nchw_bs1.rammer.nvprof.$ITERS.log --csv ./main_test
cd ../../../

cd rammer/nasnet_cifar_nchw_bs1_rammer/cuda_codegen/
./main_test > $LOG_DIR/nasnet_cifar_nchw_bs1.rammer.iter_time.$ITERS.log 2>&1
nvprof --normalized-time-unit ms --log-file $LOG_DIR/nasnet_cifar_nchw_bs1.rammer.nvprof.$ITERS.log --csv ./main_test
cd ../../../

cd rammer/alexnet_nchw_bs1_rammer/cuda_codegen/
./main_test > $LOG_DIR/alexnet_nchw_bs1.rammer.iter_time.$ITERS.log 2>&1
nvprof --normalized-time-unit ms --log-file $LOG_DIR/alexnet_nchw_bs1.rammer.nvprof.$ITERS.log --csv ./main_test
cd ../../../

cd rammer/deepspeech2_bs1_rammer/cuda_codegen/
./main_test > $LOG_DIR/deepspeech2_bs1.rammer.iter_time.$ITERS.log 2>&1
nvprof --normalized-time-unit ms --log-file $LOG_DIR/deepspeech2_bs1.rammer.nvprof.$ITERS.log --csv ./main_test
cd ../../../

cd rammer/lstm_bs1_rammer/cuda_codegen/
./main_test > $LOG_DIR/lstm_bs1.rammer.iter_time.$ITERS.log 2>&1
nvprof --normalized-time-unit ms --log-file $LOG_DIR/lstm_bs1.rammer.nvprof.$ITERS.log --csv ./main_test
cd ../../../

cd rammer/seq2seq_bs1_rammer/cuda_codegen/
./main_test > $LOG_DIR/seq2seq_bs1.rammer.iter_time.$ITERS.log 2>&1
nvprof --normalized-time-unit ms --log-file $LOG_DIR/seq2seq_bs1.rammer.nvprof.$ITERS.log --csv ./main_test
cd ../../../