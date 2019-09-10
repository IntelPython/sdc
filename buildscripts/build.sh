#!/bin/bash

source activate $CONDA_ENV

conda build --python $PYTHON_VER --override-channels -c numba/label/dev -c ehsantn -c conda-forge -c defaults ./buildscripts/hpat-conda-recipe/
