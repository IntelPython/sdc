echo on

set NUMBA_DEVELOPER_MODE=1
set NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING=1
set PYTHONFAULTHANDLER=1

python -m hpat.tests.gen_test_data
if errorlevel 1 exit 1

@rem TODO investigate root cause of NumbaPerformanceWarning
@rem http://numba.pydata.org/numba-doc/latest/user/parallel.html#diagnostics
IF "%SDC_NP_MPI%" == "" (
    python -W ignore -u -m hpat.runtests -v
    ) ELSE (
    mpiexec -localonly -n %SDC_NP_MPI% python -W ignore -u -m hpat.runtests -v)
if errorlevel 1 exit 1

REM Link check for Documentation using Sphinx's in-built linkchecker
REM sphinx-build -b linkcheck -j1 usersource _build/html
REM if errorlevel 1 exit 1

