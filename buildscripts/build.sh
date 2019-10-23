#!/bin/bash
set -ex

source activate $CONDA_ENV

conda build --python $PYTHON_VER --numpy=$NUMPY_VER -c numba -c https://metachannel.conda-forge.org/conda-forge/python,setuptools,numpy,pandas,pyarrow,arrow-cpp,boost,hdf5,h5py,mpich,wheel,coveralls,pip -c defaults --override-channels ./buildscripts/hpat-conda-recipe/
