#!/bin/bash

source activate $CONDA_ENV

# generate test data for test_io
python hpat/tests/gen_test_data.py

if [ "$RUN_COVERAGE" == "yes" ]; then
    export PYTHONPATH=.
    coverage erase
    coverage run --source=./hpat --omit ./hpat/ml/*,./hpat/xenon_ext.py,./hpat/ros.py,./hpat/cv_ext.py,./hpat/tests/gen_test_data.py -m unittest
else
    mpiexec -n $NUM_PES python -u -m unittest -v
    mpiexec -n $NUM_PES python -u -m unittest -v
    mpiexec -n $NUM_PES python -u -m unittest -v
fi
