#!/bin/bash
set -ex

# Install Miniconda
# Reference:
# https://github.com/numba/numba/blob/master/buildscripts/incremental/install_miniconda.sh

prefix="$HOME/miniconda"
if [ "$1" == "" ]; then
  echo "Conda prefix is empy; Conda will be installed to $HOME"
  echo "To set prefix use setup_conda.sh <prefix>"
else
  prefix="$1"
  echo "Conda will be installed to $prefix"
fi
# Remove miniconda if exists
if [ -d "$prefix" ]; then
  rm -rf $prefix
fi

# Download Miniconda
unamestr=`uname`
CONDA_INSTALL="conda install -q -y"
if [[ "$unamestr" == 'Linux' ]]; then
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
elif [[ "$unamestr" == 'Darwin' ]]; then
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh
else
  echo Error
fi

# Install Miniconda
chmod +x ~/miniconda.sh
~/miniconda.sh -b -p "$prefix"
export PATH=$prefix/bin:$PATH
