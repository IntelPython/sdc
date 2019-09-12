#!/bin/bash

source activate $CONDA_ENV

if [ "$BUILD_PIPELINE" == "azure" ]; then
  conda build --python $PYTHON_VER -c numba -c conda-forge -c defaults --override-channels ./buildscripts/hpat-conda-recipe/
else
  HDF5_DIR=$CONDA_PREFIX python setup.py develop
fi