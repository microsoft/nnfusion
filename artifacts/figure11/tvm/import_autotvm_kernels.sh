rm -rf autotvm_kernels
mkdir autotvm_kernels

python convert_autotvm_conv2d_log.py --input_path ../../kernel_db/autotvm_logs/resnext_nchw_tuned_conv2d.1000.log --output_path autotvm_kernels/relay_resnext_nchw_tuned_conv2d.1000.log

python convert_autotvm_conv2d_log.py --input_path ../../kernel_db/autotvm_logs/nasnet_cifar_nchw_tuned_conv2d.1000.log --output_path autotvm_kernels/relay_nasnet_cifar_nchw_tuned_conv2d.1000.log

python convert_autotvm_conv2d_log.py --input_path ../../kernel_db/autotvm_logs/alexnet_nchw_tuned_conv2d.1000.log --output_path autotvm_kernels/relay_alexnet_nchw_tuned_conv2d.1000.log

python convert_autotvm_conv2d_log.py --input_path ../../kernel_db/autotvm_logs/deepspeech2_tuned_conv2d.1000.log --output_path autotvm_kernels/relay_deepspeech2_tuned_conv2d.1000.log

python convert_autotvm_depthwise_conv2d_log.py --input_path ../../kernel_db/autotvm_logs/nasnet_cifar_nchw_tuned_depthwise_conv2d.1000.log --output_path autotvm_kernels/relay_nasnet_cifar_nchw_tuned_depthwise_conv2d.1000.log

python convert_autotvm_dense_log.py --input_path ../../kernel_db/autotvm_logs/all_tuned_topi_dense.1000.log --output_path autotvm_kernels/relay_all_tuned_dense.1000.log

cd autotvm_kernels/
cat relay_resnext_nchw_tuned_conv2d.1000.log relay_all_tuned_dense.1000.log > relay_resnext_nchw_tuned.1000.log
cat relay_nasnet_cifar_nchw_tuned_conv2d.1000.log relay_nasnet_cifar_nchw_tuned_depthwise_conv2d.1000.log relay_all_tuned_dense.1000.log > relay_nasnet_cifar_nchw_tuned.1000.log
cat relay_alexnet_nchw_tuned_conv2d.1000.log relay_all_tuned_dense.1000.log > relay_alexnet_nchw_tuned.1000.log
cat relay_deepspeech2_tuned_conv2d.1000.log relay_all_tuned_dense.1000.log > relay_deepspeech2_tuned.1000.log
cp relay_all_tuned_dense.1000.log relay_lstm_tuned.1000.log
cp relay_all_tuned_dense.1000.log relay_seq2seq_tuned.1000.log
cd ..
