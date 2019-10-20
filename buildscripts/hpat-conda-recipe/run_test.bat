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

REM Link check for Documentation using Sphinx's in-built linkchecker
REM sphinx-build -b linkcheck -j1 usersource _build/html
REM if errorlevel 1 exit 1

