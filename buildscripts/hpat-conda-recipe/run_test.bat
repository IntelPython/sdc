echo on

set NUMBA_DEVELOPER_MODE=1
set NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING=1
set PYTHONFAULTHANDLER=1

python -m hpat.tests.gen_test_data
if errorlevel 1 exit 1

IF "%HPAT_NUM_PES%" == "" (
    python -u -m hpat.runtests -v
    ) ELSE (
    mpiexec -localonly -n %HPAT_NUM_PES% python -u -m hpat.runtests -v)
if errorlevel 1 exit 1
