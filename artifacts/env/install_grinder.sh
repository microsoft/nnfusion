#!/bin/bash
cd nnfusion
mkdir build && cd build && cmake .. && make -j && cd -

cd artifacts
conda activate grinder
pip install -e .
conda deactivate
