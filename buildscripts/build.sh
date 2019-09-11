#!/bin/bash

source activate $CONDA_ENV

conda build --python $PYTHON_VER --override-channels -c numba -c conda-forge -c defaults ./buildscripts/hpat-conda-recipe/
