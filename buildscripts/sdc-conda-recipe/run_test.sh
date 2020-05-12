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
python -W ignore -u -m sdc.runtests -v sdc.tests.test_basic
python -W ignore -u -m sdc.runtests -v sdc.tests.test_d4p
python -W ignore -u -m sdc.runtests -v sdc.tests.test_dataframe
python -W ignore -u -m sdc.runtests -v sdc.tests.test_date
python -W ignore -u -m sdc.runtests -v sdc.tests.test_groupby
python -W ignore -u -m sdc.runtests -v sdc.tests.test_hiframes
python -W ignore -u -m sdc.runtests -v sdc.tests.test_hpat_jit
python -W ignore -u -m sdc.runtests -v sdc.tests.test_io
python -W ignore -u -m sdc.runtests -v sdc.tests.test_join
python -W ignore -u -m sdc.runtests -v sdc.tests.test_ml
python -W ignore -u -m sdc.runtests -v sdc.tests.test_prange_utils
python -W ignore -u -m sdc.runtests -v sdc.tests.test_rolling
python -W ignore -u -m sdc.runtests -v sdc.tests.test_sdc_numpy
python -W ignore -u -m sdc.runtests -v sdc.tests.test_series_apply
python -W ignore -u -m sdc.runtests -v sdc.tests.test_series_map
python -W ignore -u -m sdc.runtests -v sdc.tests.test_series
python -W ignore -u -m sdc.runtests -v sdc.tests.test_strings
