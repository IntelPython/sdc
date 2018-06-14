#!/bin/bash

source activate $CONDA_ENV

# generate test data for test_io
python hpat/tests/gen_test_data.py

if [ "$RUN_COVERAGE" == "yes" ]; then
    export PYTHONPATH=.
    coverage erase
    coverage run --source=./hpat --omit ./hpat/ml,./hpat/xenon_ext.py,./hpat/ros.py,./hpat/cv_ext.py,./hpat/tests/gen_test_data.py -m unittest
else
    python -m unittest
fi

mpiexec -n 2 python -m unittest
mpiexec -n 3 python -m unittest
