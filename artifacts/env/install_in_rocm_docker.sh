#!/bin/bash
cd /root/nnfusion
mkdir -p build && cd build && cmake .. && make -j

cd /root/nnfusion/artifacts
cp env/config.rocm.py ast_analyzer/utils/config.py
sed -i 's/"num_threads": 8/"num_threads": 1/g' kernel_db/test_config/get_func.py
pip install -e .
