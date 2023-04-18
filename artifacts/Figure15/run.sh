#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

conda activate baseline_tf1
./run_tf.sh
conda deactivate

conda activate baseline_jax
./run_jax.sh
conda deactivate

conda activate grinder
./run_pytorch.sh
./run_grinder.sh
conda deactivate