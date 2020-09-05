# Freeze All Evaluating Benchmark Models

In our paper, we have evaluated 6 typical DNN models:
```
resnext_nchw
nasnet_cifar_nchw
alexnet_nchw
deepspeech2
lstm
seq2seq
```

All the source code of these 6 models are located in artifacts/models/. 

## Freeze models in the model code folder to pb files

If you are using the your native environment (not in our docker), you first need to install the dependency (finish the item 3.,6.,7. in the [../README_DEPENDENCY.md](../README_DEPENDENCY.md)) before the following steps.

```bash
# freeze all the 6 models into protobuf files
cd ~/nnfusion/artifacts/frozen_models
bash freeze_models.sh
cd ..
```

After this step, all the frozen protobuf files are generated in fronzen_pbs/ folder.