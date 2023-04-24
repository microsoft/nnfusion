#!/bin/bash
cd /root/nnfusion
mkdir -p build && cd build && cmake .. && make -j

cd /root/nnfusion/artifacts
source /root/miniconda3/etc/profile.d/conda.sh
conda activate controlflow
pip install -e .
conda deactivate
