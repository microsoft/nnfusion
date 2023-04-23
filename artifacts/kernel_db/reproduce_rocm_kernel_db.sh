#!/bin/bash

python3 manual_kernels.py --device ROCM_GPU
python3 roller_rocm_kernels.py --reproduce
python3 ansor_rocm_kernels.py