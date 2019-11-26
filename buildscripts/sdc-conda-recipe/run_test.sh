#!/bin/bash

set -ex

export NUMBA_DEVELOPER_MODE=1
export NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING=1
export PYTHONFAULTHANDLER=1

python -m sdc.tests.gen_test_data

#Link check for Documentation using Sphinx's in-built linkchecker
#sphinx-build -b linkcheck -j1 usersource _build/html

# TODO investigate root cause of NumbaPerformanceWarning
# http://numba.pydata.org/numba-doc/latest/user/parallel.html#diagnostics
if [ -z "$SDC_NP_MPI" ]; then
  python -W ignore -u -m sdc.runtests -v
else
  mpiexec -n $SDC_NP_MPI python -W ignore -u -m sdc.runtests -v
fi
