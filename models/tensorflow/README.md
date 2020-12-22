# TensorFlow Frozen Model Zoo

NNFusion supports compiling Tensorflow through taking its frozen format (i.e., a protobuf file) as input. For more information about how to freeze a TensorFlow model into a frozen format, please refer to [Freeze TensorFlow models](nnf_tf_freezer/README.md).

This page lists some commonly-used frozen models that are well tested with NNFusion. These models contain typical DNN architecures such as CNN, RNN, Transformer, etc., and cover the most common DNN domains including image, NLP and speech.


| model        | type        | format | TF version | download link |
| -----------  | ----------- | -------| ---------  | ------------- |              
| AlexNet      | inference   | frozen | 1.14 | [frozen_alexnet_infer_batch_1.const_folded.pb](https://nnfusion.blob.core.windows.net/models/tensorflow/frozen_alexnet_infer_batch_1.const_folded.pb)  |
| VGG11        | inference   | frozen | 1.14 | [frozen_vgg11_infer_batch_1.const_folded.pb](https://nnfusion.blob.core.windows.net/models/tensorflow/frozen_vgg11_infer_batch_1.const_folded.pb)    |
| ResNet50     | inference   | frozen | 1.14 | [frozen_resnet50_infer_batch_1.const_folded.pb](https://nnfusion.blob.core.windows.net/models/tensorflow/frozen_resnet50_infer_batch_1.const_folded.pb) |
| Inception_v3 | inference   | frozen | 1.14 | [frozen_inception3_infer_batch_1.const_folded.pb](https://nnfusion.blob.core.windows.net/models/tensorflow/frozen_inception3_infer_batch_1.const_folded.pb)
| LSTM-L10-L100-H256 | inference   | frozen | 1.14 | [frozen_lstm_infer_batch_1.const_folded.pb](https://nnfusion.blob.core.windows.net/models/tensorflow/frozen_lstm_infer_batch_1.const_folded.pb)
| LSTM-L8-S8-H256 | inference | frozen | 1.14 | [frozen_lstm_l8s8h256_bs1.pb](https://nnfusion.blob.core.windows.net/models/tensorflow/frozen_lstm_l8s8h256_bs1.pb) | 
| BERT_large | inference | frozen | 1.14 | [frozen_bert_large.const_folded.pb](https://nnfusion.blob.core.windows.net/models/tensorflow/frozen_bert_large.const_folded.pb) | 
| BERT_large_L2 | inference | frozen | 1.14 | [frozen_bert_large_layer_2.const_folded.pb](https://nnfusion.blob.core.windows.net/models/tensorflow/frozen_bert_large_layer_2.const_folded.pb) | 

## Usage Example: Compile LSTM model on CUDA

Prerequisite: We assume you already build and install NNFusion compiler folloing the [Build Guide](https://github.com/microsoft/nnfusion/wiki/Build-Guide).

Take LSTM-L8-S8-H256 model as an example, you can just download the model:

```
wget https://nnfusion.blob.core.windows.net/models/tensorflow/tensorflow/frozen_lstm_l8s8h256_bs1.pb
```

Then compile the model with NNFusion (we assume you have a CUDA envrioment):
```
NNFUSION_INSTALL_PATH/nnfusion tensorflow/frozen_lstm_l8s8h256_bs1.pb --format tensorflow -fdefault_device CUDA
```

If everything goes smoothly, you will see the generated full project code for the LSTM model under: nnfusion_rt/cuda_codegen/. 
Then you can build the generated project and test performance through:
```
cd nnfusion_rt/cuda_codegen/
cmake . && make
./main_test
```
The test will iterattively run the mdoel for 100 times and calculate the average latency. The example logs will look like:
```
Result_2261_0: 
8.921492e-03 1.182089e-02 8.937407e-03 7.932202e-03 1.574193e-02 3.844392e-03 -1.505094e-02 -1.112035e-02 5.026605e-03 -8.032203e-03  .. (size = 256, ends with 1.357487e-02);
Iteration time 2.990464 ms
...
Iteration time 2.700096 ms
Iteration time 2.702432 ms
Summary: [min, max, mean] = [2.690368, 6.759712, 2.918306] ms
```
You can see the avearge latency on a P100 GPU is about 2.918306 ms. Note that, this just adopted the basic optimization in NNFusion, to further optimize this model's latency to less than 1 ms, please follow the tutorial of our recent technique called BlockFusion in [Rammer OSDI Tutorial](https://github.com/microsoft/nnfusion/blob/osdi20_artifact/artifacts/get_started_tutorial/README_GET_STARTED.md).


