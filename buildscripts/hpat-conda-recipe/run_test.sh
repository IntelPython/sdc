#!/bin/bash

set -ex

export NUMBA_DEVELOPER_MODE=1
export NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING=1
export PYTHONFAULTHANDLER=1

python -m hpat.tests.gen_test_data

if [ "$HPAT_WHEELS" == "True" ]; then
  conda remove -y hpat
  pip install ${HPAT_WHEELS_DIR}/hpat*.whl
  python -c "import hpat"
fi

if [ "$HPAT_RUN_COVERAGE" == "True" ]; then
  coverage erase
  coverage run --source=${PREFIX}/lib/python3.7/site-packages/hpat --omit ${PREFIX}/lib/python3.7/site-packages/hpat/ml/*,${PREFIX}/lib/python3.7/site-packages/hpat/xenon_ext.py,${PREFIX}/lib/python3.7/site-packages/hpat/ros.py,${PREFIX}/lib/python3.7/site-packages/hpat/cv_ext.py,${PREFIX}/lib/python3.7/site-packages/hpat/tests/* ${PREFIX}/lib/python3.7/site-packages/hpat/runtests.py
  coveralls -v
else
  #Link check for Documentation using Sphinx's in-built linkchecker
  #sphinx-build -b linkcheck -j1 usersource _build/html

  # TODO investigate root cause of NumbaPerformanceWarning
  # http://numba.pydata.org/numba-doc/latest/user/parallel.html#diagnostics
  if [ -z "$HPAT_NUM_PES" ]; then
    python -W ignore -u -m hpat.runtests -v
  else
    mpiexec -n $HPAT_NUM_PES python -W ignore -u -m hpat.runtests -v
  fi
fi
