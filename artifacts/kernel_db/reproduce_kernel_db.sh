#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh

conda activate kerneldb

python3 manual_kernels.py
python3 roller_kernels.py --reproduce
python3 autotvm_matmul_kernels.py

for i in {0..11}; do
    python3 ansor_kernels.py --tid $i --trial 0 --inject
done

conda deactivate