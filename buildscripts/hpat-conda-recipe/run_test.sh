#!/bin/bash

set -e

export NUMBA_DEVELOPER_MODE=1
export NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING=1
export PYTHONFAULTHANDLER=1

python -m hpat.tests.gen_test_data

if [ "$HPAT_RUN_COVERAGE" == "True" ]; then
  export PYTHONPATH=.
  coverage erase
  coverage run --source=./hpat --omit ./hpat/ml/*,./hpat/xenon_ext.py,./hpat/ros.py,./hpat/cv_ext.py,./hpat/tests/* ./hpat/runtests.py
else
  # TODO investigate root cause of NumbaPerformanceWarning
  # http://numba.pydata.org/numba-doc/latest/user/parallel.html#diagnostics
  if [ -z "$HPAT_NUM_PES" ]; then
    python -W ignore -u -m hpat.runtests -v
  else
    mpiexec -n $HPAT_NUM_PES python -W ignore -u -m hpat.runtests -v
  fi
fi
