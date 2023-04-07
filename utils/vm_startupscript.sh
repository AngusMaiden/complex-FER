#!/bin/bash

# Update all linux packages.
sudo apt -y update && sudo apt -y upgrade

# Install NVIDIA driver
sudo apt install build-essential -y
wget https://uk.download.nvidia.com/tesla/470.129.06/NVIDIA-Linux-x86_64-470.129.06.run
sudo sh NVIDIA-Linux-x86_64-470.129.06.run

# Delete downloaded .run file
rm NVIDIA-Linux-x86_64-470.129.06.run

# Downloads and installs miniconda silently
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda

export PATH="$HOME/miniconda/bin:$PATH"
conda init

# Install CUDA and cuDNN
conda install -c conda-forge -y cudatoolkit=11.2 cudnn=8.1.0
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Activate conda env
conda activate

# Install python libraries
pip install -r requirements.txt

# Clone repository
git clone https://github.com/AngusMaiden/complex-FER.git

# Reboot required
sudo reboot