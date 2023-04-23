#!/bin/bash

mkdir -p ../reproduce_results/plot

source ~/miniconda3/etc/profile.d/conda.sh
conda activate controlflow
python3 figure2.py
python3 figure14.py
python3 figure15.py
python3 figure16.py
python3 figure17.py
python3 figure18.py
python3 figure19.py
python3 figure20.py
conda deactivate
