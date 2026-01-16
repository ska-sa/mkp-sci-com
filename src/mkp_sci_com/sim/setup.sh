#!/bin/bash

# Setup Python environment
conda create -y -n mkp python=3.12 pip
conda activate mkp
pip install stimela cult-cargo simms eidos casaconfig casadata casatasks casatools

# Ensure that casadata directory is populated
mkdir -p $HOME/.casa/dat

# Ensure that singularity/apptainer build directory is set to $HOME/tmp
# and the directory is populated as users usually do not have disk quota 
# to the the default /tmp 
mkdir -p $HOME/tmp

cat << 'MY_BASHRC_BLOCK' >> ~/.bashrc

# Move singularity/apptainer build directory from the default /tmp"
export SINGULARITY_TMPDIR=$HOME/tmp
MY_BASHRC_BLOCK
source ~/.bashrc

# Build singularity image for meqtree-pipeliner
# The image will be stored in ~/.singularity/
stimela build cultcargo::meqtree-pipeliner