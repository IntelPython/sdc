echo on

set NUMBA_DEVELOPER_MODE=1
set NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING=1
set PYTHONFAULTHANDLER=1

python -m sdc.tests.gen_test_data
if errorlevel 1 exit 1

@rem TODO investigate root cause of NumbaPerformanceWarning
@rem http://numba.pydata.org/numba-doc/latest/user/parallel.html#diagnostics
python -W ignore -u -m sdc.runtests -v sdc.tests.test_basic
if errorlevel 1 exit 1
python -W ignore -u -m sdc.runtests -v sdc.tests.test_d4p
if errorlevel 1 exit 1
python -W ignore -u -m sdc.runtests -v sdc.tests.test_dataframe
if errorlevel 1 exit 1
python -W ignore -u -m sdc.runtests -v sdc.tests.test_date
if errorlevel 1 exit 1
python -W ignore -u -m sdc.runtests -v sdc.tests.test_groupby
if errorlevel 1 exit 1
python -W ignore -u -m sdc.runtests -v sdc.tests.test_hiframes
if errorlevel 1 exit 1
python -W ignore -u -m sdc.runtests -v sdc.tests.test_hpat_jit
if errorlevel 1 exit 1
python -W ignore -u -m sdc.runtests -v sdc.tests.test_io
if errorlevel 1 exit 1
python -W ignore -u -m sdc.runtests -v sdc.tests.test_join
if errorlevel 1 exit 1
python -W ignore -u -m sdc.runtests -v sdc.tests.test_ml
if errorlevel 1 exit 1
python -W ignore -u -m sdc.runtests -v sdc.tests.test_prange_utils
if errorlevel 1 exit 1
python -W ignore -u -m sdc.runtests -v sdc.tests.test_rolling
if errorlevel 1 exit 1
python -W ignore -u -m sdc.runtests -v sdc.tests.test_sdc_numpy
if errorlevel 1 exit 1
python -W ignore -u -m sdc.runtests -v sdc.tests.test_series_apply
if errorlevel 1 exit 1
python -W ignore -u -m sdc.runtests -v sdc.tests.test_series_map
if errorlevel 1 exit 1
python -W ignore -u -m sdc.runtests -v sdc.tests.test_series
if errorlevel 1 exit 1
python -W ignore -u -m sdc.runtests -v sdc.tests.test_strings
if errorlevel 1 exit 1

REM Link check for Documentation using Sphinx's in-built linkchecker
REM sphinx-build -b linkcheck -j1 usersource _build/html
REM if errorlevel 1 exit 1
