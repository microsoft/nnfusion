#!/bin/bash
cd /root/nnfusion
mkdir -p build && cd build && cmake .. && make -j

cd /root/nnfusion/artifacts
pip install -e .
