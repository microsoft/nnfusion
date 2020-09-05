# Experiment #2: Performance with different batch sizes

This experiment is used to show the performance comparison on two representative CNN and RNN models, i.e., ResNeXt and LSTM-TC, with batch sizes of 4 and 16, and to reproduce the results in Figure 11 of our origianl paper.

## Requirements

If you are using our Docker container environment, you can just skip this step. Otherwise, you need to finish the item 1-6, 8-12 in the [../README_DEPENDENCY.md](../README_DEPENDENCY.md) and all the items in [../README_FREEZE_MODEL.md](../README_FREEZE_MODEL.md).

## Reproduce results
Use NNFusion to compile all the frozen models:
```
cd /root/nnfusion/artifacts/figure11
bash codegen_and_build.sh
```
Run all baselines and NNFusion on all the benchmarks, the corresponding output logs are generated in individual folders. 
Note that, this will take a relativley long time as each of the running needs to iterative for 1000 times.
```
bash run_all.sh
```
Process all the logs and generate the final performance numbers in a Gnuplot input format:
```
python process_log.py
```
Plot the end-to-end comparision figure (i.g., Figure 11). 
```
cd reproduce_result/
gnuplot gpu1_batch_cuda_multifig.plt
```
Fianlly, in the reproduce_result folder, you will see the "figure11_paper.pdf".
To compare with paper results, we put the paper data and the same plotting script under the *paper_result* folder.

### End-to-end script
All the above steps can be exected by the below single script:
```
bash reproduce_figure11.sh
```