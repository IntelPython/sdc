#!/bin/bash

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
# Install conda-build
$CONDA_INSTALL conda-build

# Environment for Travis build
if [ "$BUILD_PIPELINE" == "travis" ]; then
  $CONDA_INSTALL pyarrow
  # install compilers
  if [[ "$unamestr" == 'Linux' ]]; then
    $CONDA_INSTALL gcc_linux-64 gxx_linux-64 gfortran_linux-64
  elif [[ "$unamestr" == 'Darwin' ]]; then
    $CONDA_INSTALL clang_osx-64 clangxx_osx-64 gfortran_osx-64
  else
    echo "Error in compiler install"
  fi
  $CONDA_INSTALL mpich mpi -c conda-forge --no-deps
  $CONDA_INSTALL -c numba numba
  $CONDA_INSTALL libgfortran
  $CONDA_INSTALL h5py
  $CONDA_INSTALL coveralls
fi