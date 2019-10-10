#!/bin/bash

source activate $CONDA_ENV

conda build --python $PYTHON_VER -c numba -c conda-forge -c defaults --override-channels ./buildscripts/hpat-conda-recipe/
