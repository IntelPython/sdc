#!/bin/bash

source activate $CONDA_ENV

if [ "$HPAT_RUN_COVERAGE" == "yes" ]; then	
    export PYTHONPATH=.
    coverage erase
    coverage run --source=./hpat --omit ./hpat/ml/*,./hpat/xenon_ext.py,./hpat/ros.py,./hpat/cv_ext.py,./hpat/tests/* -m hpat.runtests
else
    if [ -z "$HPAT_NUM_PES" ]; then
      python -W ignore -u -m hpat.runtests -v
    else
      mpiexec -n $HPAT_NUM_PES python -W ignore -u -m hpat.runtests -v
   fi
fi