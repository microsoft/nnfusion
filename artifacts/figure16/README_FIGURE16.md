# Experiment #4: Comparison of GPU scheduling overhead

Rammer's techniques can effectively reduce scheduling overhead. 
This experiment is used to evaluate the run-time scheduling overhead by comparing Rammer with both TF and RammerBase.

## Requirements

If you are using our Docker container environment, you can just skip this step. Otherwise, you need to finish the item 1-6, 8-12 in the [../README_DEPENDENCY.md](../README_DEPENDENCY.md) and all the items in [../README_FREEZE_MODEL.md](../README_FREEZE_MODEL.md).

## Reproduce results
Use NNFusion to compile all the frozen models:
```
cd /root/nnfusion/artifacts/figure16
bash codegen_and_build.sh
```
Run all baselines and NNFusion on all the benchmarks, the corresponding output logs are generated in individual folders. 
```
bash run_all.sh
```
Process all the logs and generate the final performance numbers in a Gnuplot input format:
```
python process_log.py
```
Plot the end-to-end GPU scheduling overhead figure (i.g., Figure 14). 
```
cd reproduce_result/
gnuplot gpu1_gpu_schedoverhead_cuda.plt
```
Fianlly, in the reproduce_result folder, you will see the "figure16_paper.pdf".
To compare with paper results, we put the paper data and the same plotting script under the *paper_result* folder.

### End-to-end script
All the above steps can be exected by the below single script:
```
bash reproduce_figure16.sh
```