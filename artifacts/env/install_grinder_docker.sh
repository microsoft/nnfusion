#!/bin/bash
cd /root/nnfusion
mkdir build && cd build && cmake .. && make -j

cd /root/nnfusion/artifacts
source /root/miniconda3/etc/profile.d/conda.sh
conda activate grinder
pip install -e .
conda deactivate
