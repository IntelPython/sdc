#!/bin/bash

source activate $CONDA_ENV

conda build --python $PYTHON_VER -c numba -c ehsantn -c conda-forge -c defaults ./buildscripts/hpat-conda-recipe/
