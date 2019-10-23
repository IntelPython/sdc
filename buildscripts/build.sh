#!/bin/bash
set -ex

source activate $CONDA_ENV

conda build --python $PYTHON_VER --numpy=$NUMPY_VER -c numba -c conda-forge -c defaults --override-channels ./buildscripts/hpat-conda-recipe/
