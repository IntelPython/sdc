#!/bin/bash
set -e

# Install Miniconda
# Reference:
# https://github.com/numba/numba/blob/master/buildscripts/incremental/install_miniconda.sh

echo "Install Miniconda3"

prefix="$HOME/miniconda3"
if [ "$1" == "" ]; then
  echo "Conda prefix is empy; Conda will be installed to $HOME"
  echo "To set prefix use setup_conda.sh <prefix>"
else
  prefix="$1"
fi
echo "Conda will be installed to $prefix"

# Remove miniconda if exists
echo "Remove $prefix"
if [ -d "$prefix" ]; then
  rm -rf $prefix
fi

# Download Miniconda
echo "Download Miniconda3 installer to ~/miniconda.sh"
unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
elif [[ "$unamestr" == 'Darwin' ]]; then
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh
else
  echo Error
fi

# Install Miniconda
echo "Install Miniconda3 to $prefix"
chmod +x ~/miniconda.sh
~/miniconda.sh -b -p "$prefix"
echo "Done"
