# Experiment #5: Interplay of Intra and Inter Operator Scheduling

Rammer enables scheduling policies to optimize the interplay of intra and inter operator scheduling, instead of just focusing on making individual operators fast.
This is implemented through selecting appropriate kernels for each operator. 
We evaluate the effect of such scheduling by using two sets of kernels: the fastest kernels only for each individual operator, and the kernels selected by Rammer's scheduling policy.
This experiment is usedto reproduce the results in Figure 15 of our origianl paper.

## Requirements

If you are using our Docker container environment, you can just skip this step. Otherwise, you need to finish the item 1-6, 8-12 in the [../README_DEPENDENCY.md](../README_DEPENDENCY.md) and all the items in [../README_FREEZE_MODEL.md](../README_FREEZE_MODEL.md).

## Reproduce results
Use NNFusion to compile all the frozen models:
```
cd /root/nnfusion/artifacts/figure15
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
Plot the end-to-end comparision figure (i.g., Figure 15). 
```
cd reproduce_result/
gnuplot gpu1_interplay_cuda_multifig.plt
```
Fianlly, in the reproduce_result folder, you will see the "figure15_paper.pdf".
To compare with paper results, we put the paper data and the same plotting script under the *paper_result* folder.

### End-to-end script
All the above steps can be exected by the below single script:
```
bash reproduce_figure15.sh
```