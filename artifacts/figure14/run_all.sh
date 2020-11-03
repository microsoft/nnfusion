# tf
MODEL_DIR=$ARTIFACTS_HOME/models
LOG_DIR=$ARTIFACTS_HOME/figure14/logs
STEP=3
WARMUP=1
ITERS=$STEP+$WARMUP

rm -rf $LOG_DIR
mkdir $LOG_DIR


for BS in 1
do
    cd $MODEL_DIR/alexnet_nchw
    nvprof --csv --print-gpu-trace --metrics sm_efficiency python alexnet_inference.py --num_iter $STEP --batch_size $BS --warmup $WARMUP --parallel 1 > $LOG_DIR/alexnet_nchw_bs$BS.tf.sm_efficiency.$ITERS.log 2>&1
    nvprof --csv --print-gpu-trace python alexnet_inference.py --num_iter $STEP --batch_size $BS --warmup $WARMUP --parallel 1 > $LOG_DIR/alexnet_nchw_bs$BS.tf.kernel_trace.$ITERS.log 2>&1


    cd $MODEL_DIR/deepspeech2
    nvprof --csv --print-gpu-trace --metrics sm_efficiency python deep_speech_inference.py --num_iter $STEP --batch_size $BS --warmup $WARMUP --parallel 1 > $LOG_DIR/deepspeech2_bs$BS.tf.sm_efficiency.$ITERS.log 2>&1
    nvprof --csv --print-gpu-trace python deep_speech_inference.py --num_iter $STEP --batch_size $BS --warmup $WARMUP --parallel 1 > $LOG_DIR/deepspeech2_bs$BS.tf.kernel_trace.$ITERS.log 2>&1


    cd $MODEL_DIR/lstm
    nvprof --csv --print-gpu-trace --metrics sm_efficiency python lstm_inference.py --num_iter $STEP --batch_size $BS --warmup $WARMUP --parallel 1 > $LOG_DIR/lstm_bs$BS.tf.sm_efficiency.$ITERS.log 2>&1
    nvprof --csv --print-gpu-trace python lstm_inference.py --num_iter $STEP --batch_size $BS --warmup $WARMUP --parallel 1 > $LOG_DIR/lstm_bs$BS.tf.kernel_trace.$ITERS.log 2>&1


    cd $MODEL_DIR/nasnet_cifar_nchw
    nvprof --csv --print-gpu-trace --metrics sm_efficiency python nasnet_cifar_inference.py --num_iter $STEP --batch_size $BS --warmup $WARMUP --parallel 1 > $LOG_DIR/nasnet_cifar_nchw_bs$BS.tf.sm_efficiency.$ITERS.log 2>&1
    nvprof --csv --print-gpu-trace python nasnet_cifar_inference.py --num_iter $STEP --batch_size $BS --warmup $WARMUP --parallel 1 > $LOG_DIR/nasnet_cifar_nchw_bs$BS.tf.kernel_trace.$ITERS.log 2>&1


    cd $MODEL_DIR/resnext_nchw
    nvprof --csv --print-gpu-trace --metrics sm_efficiency python resnext_inference.py --num_iter $STEP --batch_size $BS --warmup $WARMUP --parallel 1 > $LOG_DIR/resnext_nchw_bs$BS.tf.sm_efficiency.$ITERS.log 2>&1
    nvprof --csv --print-gpu-trace python resnext_inference.py --num_iter $STEP --batch_size $BS --warmup $WARMUP --parallel 1 > $LOG_DIR/resnext_nchw_bs$BS.tf.kernel_trace.$ITERS.log 2>&1


    cd $MODEL_DIR/seq2seq
    nvprof --csv --print-gpu-trace --metrics sm_efficiency python seq2seq_inference.py --num_iter $STEP --batch_size $BS --warmup $WARMUP --parallel 1 > $LOG_DIR/seq2seq_bs$BS.tf.sm_efficiency.$ITERS.log 2>&1
    nvprof --csv --print-gpu-trace python seq2seq_inference.py --num_iter $STEP --batch_size $BS --warmup $WARMUP --parallel 1 > $LOG_DIR/seq2seq_bs$BS.tf.kernel_trace.$ITERS.log 2>&1
done


cd $ARTIFACTS_HOME/figure14/

# profile tensorrt
cd trt/

nvprof --csv --print-gpu-trace --metrics sm_efficiency python tf_trt_run_frozen.py --warmup $WARMUP --num_iter $STEP --model ResNextNchw --model_path ../../frozen_models/frozen_pbs/frozen_resnext_nchw_infer_bs1.const_folded.pb > ../logs/resnext_nchw_bs1.trt.sm_efficiency.$ITERS.log 2>&1
nvprof --csv --print-gpu-trace python tf_trt_run_frozen.py --warmup $WARMUP --num_iter $STEP --model ResNextNchw --model_path ../../frozen_models/frozen_pbs/frozen_resnext_nchw_infer_bs1.const_folded.pb > ../logs/resnext_nchw_bs1.trt.kernel_trace.$ITERS.log 2>&1

nvprof --csv --print-gpu-trace --metrics sm_efficiency python tf_trt_run_frozen.py --warmup $WARMUP --num_iter $STEP --model NasnetCifarNchw --model_path ../../frozen_models/frozen_pbs/frozen_nasnet_cifar_nchw_infer_bs1.const_folded.pb > ../logs/nasnet_cifar_nchw_bs1.trt.sm_efficiency.$ITERS.log 2>&1
nvprof --csv --print-gpu-trace python tf_trt_run_frozen.py --warmup $WARMUP --num_iter $STEP --model NasnetCifarNchw --model_path ../../frozen_models/frozen_pbs/frozen_nasnet_cifar_nchw_infer_bs1.const_folded.pb > ../logs/nasnet_cifar_nchw_bs1.trt.kernel_trace.$ITERS.log 2>&1

nvprof --csv --print-gpu-trace --metrics sm_efficiency python tf_trt_run_frozen.py --warmup $WARMUP --num_iter $STEP --model AlexnetNchw --model_path ../../frozen_models/frozen_pbs/frozen_alexnet_nchw_infer_bs1.const_folded.pb > ../logs/alexnet_nchw_bs1.trt.sm_efficiency.$ITERS.log 2>&1
nvprof --csv --print-gpu-trace python tf_trt_run_frozen.py --warmup $WARMUP --num_iter $STEP --model AlexnetNchw --model_path ../../frozen_models/frozen_pbs/frozen_alexnet_nchw_infer_bs1.const_folded.pb > ../logs/alexnet_nchw_bs1.trt.kernel_trace.$ITERS.log 2>&1

cd deepspeech2_native/deepspeech/
make -j
nvprof --csv --print-gpu-trace --metrics sm_efficiency bin/deepspeech > $LOG_DIR/deepspeech2_bs1.trt.sm_efficiency.$ITERS.log 2>&1
nvprof --csv --print-gpu-trace bin/deepspeech > $LOG_DIR/deepspeech2_bs1.trt.kernel_trace.$ITERS.log 2>&1
cd ../..

cd lstm_native/lstm/
make -j
nvprof --csv --print-gpu-trace --metrics sm_efficiency bin/lstm > $LOG_DIR/lstm_bs1.trt.sm_efficiency.$ITERS.log 2>&1
nvprof --csv --print-gpu-trace bin/lstm > $LOG_DIR/lstm_bs1.trt.kernel_trace.$ITERS.log 2>&1
cd ../..

cd seq2seq_native/seq2seq/
make -j
nvprof --csv --print-gpu-trace --metrics sm_efficiency bin/seq2seq > $LOG_DIR/seq2seq_bs1.trt.sm_efficiency.$ITERS.log 2>&1
nvprof --csv --print-gpu-trace bin/seq2seq > $LOG_DIR/seq2seq_bs1.trt.kernel_trace.$ITERS.log 2>&1
cd ../..

cd ..

# profile rammer_base
cd rammer_base/

cd resnext_nchw_bs1_rammerbase/cuda_codegen
nvprof --csv --print-gpu-trace --metrics sm_efficiency ./main_test > ../../../logs/resnext_nchw_bs1.rammerbase.sm_efficiency.log 2>&1
nvprof --csv --print-gpu-trace ./main_test > ../../../logs/resnext_nchw_bs1.rammerbase.kernel_trace.log 2>&1
cd ../..

cd nasnet_cifar_nchw_bs1_rammerbase/cuda_codegen
nvprof --csv --print-gpu-trace --metrics sm_efficiency ./main_test > ../../../logs/nasnet_cifar_nchw_bs1.rammerbase.sm_efficiency.log 2>&1
nvprof --csv --print-gpu-trace ./main_test > ../../../logs/nasnet_cifar_nchw_bs1.rammerbase.kernel_trace.log 2>&1
cd ../..

cd alexnet_nchw_bs1_rammerbase/cuda_codegen
nvprof --csv --print-gpu-trace --metrics sm_efficiency ./main_test > ../../../logs/alexnet_nchw_bs1.rammerbase.sm_efficiency.log 2>&1
nvprof --csv --print-gpu-trace ./main_test > ../../../logs/alexnet_nchw_bs1.rammerbase.kernel_trace.log 2>&1
cd ../..

cd deepspeech2_bs1_rammerbase/cuda_codegen
nvprof --csv --print-gpu-trace --metrics sm_efficiency ./main_test > ../../../logs/deepspeech2_bs1.rammerbase.sm_efficiency.log 2>&1
nvprof --csv --print-gpu-trace ./main_test > ../../../logs/deepspeech2_bs1.rammerbase.kernel_trace.log 2>&1
cd ../..

cd lstm_bs1_rammerbase/cuda_codegen
nvprof --csv --print-gpu-trace --metrics sm_efficiency ./main_test > ../../../logs/lstm_bs1.rammerbase.sm_efficiency.log 2>&1
nvprof --csv --print-gpu-trace ./main_test > ../../../logs/lstm_bs1.rammerbase.kernel_trace.log 2>&1
cd ../..

cd seq2seq_bs1_rammerbase/cuda_codegen
nvprof --csv --print-gpu-trace --metrics sm_efficiency ./main_test > ../../../logs/seq2seq_bs1.rammerbase.sm_efficiency.log 2>&1
nvprof --csv --print-gpu-trace ./main_test > ../../../logs/seq2seq_bs1.rammerbase.kernel_trace.log 2>&1
cd ../..

cd ..

# profile rammer
cd rammer/

cd resnext_nchw_bs1_rammer/cuda_codegen
nvprof --csv --print-gpu-trace --metrics sm_efficiency ./main_test > ../../../logs/resnext_nchw_bs1.rammer.sm_efficiency.log 2>&1
nvprof --csv --print-gpu-trace ./main_test > ../../../logs/resnext_nchw_bs1.rammer.kernel_trace.log 2>&1
cd ../..

cd nasnet_cifar_nchw_bs1_rammer/cuda_codegen
nvprof --csv --print-gpu-trace --metrics sm_efficiency ./main_test > ../../../logs/nasnet_cifar_nchw_bs1.rammer.sm_efficiency.log 2>&1
nvprof --csv --print-gpu-trace ./main_test > ../../../logs/nasnet_cifar_nchw_bs1.rammer.kernel_trace.log 2>&1
cd ../..

cd alexnet_nchw_bs1_rammer/cuda_codegen
nvprof --csv --print-gpu-trace --metrics sm_efficiency ./main_test > ../../../logs/alexnet_nchw_bs1.rammer.sm_efficiency.log 2>&1
nvprof --csv --print-gpu-trace ./main_test > ../../../logs/alexnet_nchw_bs1.rammer.kernel_trace.log 2>&1
cd ../..

cd deepspeech2_bs1_rammer/cuda_codegen
nvprof --csv --print-gpu-trace --metrics sm_efficiency ./main_test > ../../../logs/deepspeech2_bs1.rammer.sm_efficiency.log 2>&1
nvprof --csv --print-gpu-trace ./main_test > ../../../logs/deepspeech2_bs1.rammer.kernel_trace.log 2>&1
cd ../..

cd lstm_bs1_rammer/cuda_codegen
nvprof --csv --print-gpu-trace --metrics sm_efficiency ./main_test > ../../../logs/lstm_bs1.rammer.sm_efficiency.log 2>&1
nvprof --csv --print-gpu-trace ./main_test > ../../../logs/lstm_bs1.rammer.kernel_trace.log 2>&1
cd ../..

cd seq2seq_bs1_rammer/cuda_codegen
nvprof --csv --print-gpu-trace --metrics sm_efficiency ./main_test > ../../../logs/seq2seq_bs1.rammer.sm_efficiency.log 2>&1
nvprof --csv --print-gpu-trace ./main_test > ../../../logs/seq2seq_bs1.rammer.kernel_trace.log 2>&1
cd ../..

cd ..