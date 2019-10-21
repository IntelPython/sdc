#!/bin/bash
set -ex

# Install Miniconda
# Reference:
# https://github.com/numba/numba/blob/master/buildscripts/incremental/install_miniconda.sh

# Download
unamestr=`uname`
CONDA_INSTALL="conda install -q -y"
if [[ "$unamestr" == 'Linux' ]]; then
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
elif [[ "$unamestr" == 'Darwin' ]]; then
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
else
  echo Error
fi
# Install
chmod +x miniconda.sh
./miniconda.sh -b
export PATH=$HOME/miniconda3/bin:$PATH

# Create conda env
conda create -n $CONDA_ENV -q -y python=$PYTHON_VER
source activate $CONDA_ENV
# Install conda-build and numpy with pycodestyle (required for style check)
$CONDA_INSTALL conda-build numpy pycodestyle clang pip -c conda-forge
if [ "$HPAT_CHECK_STYLE" == "True" ]; then
  pip install clang
fi
