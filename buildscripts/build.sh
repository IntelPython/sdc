#!/bin/bash

source activate $CONDA_ENV

if [ "$CONDA_RUN_TEST" == "False" ]; then
  conda build --python $PYTHON_VER --no-test -c numba -c conda-forge -c defaults --override-channels ./buildscripts/hpat-conda-recipe/
else
  conda build --python $PYTHON_VER -c numba -c conda-forge -c defaults --override-channels ./buildscripts/hpat-conda-recipe/
