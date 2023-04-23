#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

# conda activate controlflow
# ./run_pytorch.sh
# conda deactivate

# conda activate baseline_tf1
# ./run_tf.sh
# conda deactivate

conda activate baseline_jax
./run_jax.sh
conda deactivate

conda activate controlflow
./run_sys.sh
conda deactivate
