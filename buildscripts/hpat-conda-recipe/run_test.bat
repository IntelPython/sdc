set NUMBA_DEVELOPER_MODE=1
set NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING=1
set PYTHONFAULTHANDLER=1

mpiexec -localonly -n 1 python -m hpat.tests.gen_test_data
if errorlevel 1 exit 1

mpiexec -localonly -n 1 python -u -m hpat.runtests -v
if errorlevel 1 exit 1

mpiexec -localonly -n 2 python -u -m hpat.runtests -v
if errorlevel 1 exit 1

mpiexec -localonly -n 3 python -u -m hpat.runtests -v
if errorlevel 1 exit 1
