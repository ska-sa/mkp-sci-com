#!/bin/bash

# Setup Python environment
conda create -n mkp python=3.12 pip
conda activate mkp
pip install stimela cult-cargo simms

# Ensure that singularity exist
# if not
# $(which singularity)
# raise error and exist

# Build singularity image for meqtree-pipeliner
# The image will be stored in ~/.singularity/
stimela build cultcargo::meqtree-pipeliner