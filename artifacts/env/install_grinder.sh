#!/bin/bash
cd /root/nnfusion
mkdir build && cd build && cmake .. && make -j

cd /root/nnfusion/artifacts
conda activate grinder
pip install -e .
conda deactivate
