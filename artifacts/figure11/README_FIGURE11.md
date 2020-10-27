# Experiment #1: End-to-end comparation with the state-of-the-arts

This experiment is used to demonstrate the end-to-end efficiency of Rammer by comparing with TensorFlow (TF), TensorFlow-XLA (TF-XLA), TVM and TensorRT (TF-TRT), and to reproduce the results in Figure 11 of our origianl paper.

## Requirements

If you are using our Docker container environment, you can just skip this step. Otherwise, you need to finish the item 1-6, 8-12 in the [../README_DEPENDENCY.md](../README_DEPENDENCY.md) and all the items in [../README_FREEZE_MODEL.md](../README_FREEZE_MODEL.md).

## Reproduce results
Use NNFusion to compile all the frozen models:
```
cd /root/nnfusion/artifacts/figure11
source ../.profile
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
gnuplot gpu1_e2e_cuda_multifig.plt
```
Fianlly, in the reproduce_result folder, you will see the "figure11_paper.pdf". To compare with paper results, we put the paper data and the same plotting script under the *paper_result* folder.

### End-to-end script
All the above steps can be exected by the below single script:
```
bash reproduce_figure11.sh
```