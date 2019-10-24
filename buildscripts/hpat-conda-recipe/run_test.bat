echo on

set NUMBA_DEVELOPER_MODE=1
set NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING=1
set PYTHONFAULTHANDLER=1

python -m hpat.tests.gen_test_data
if errorlevel 1 exit 1

IF "%HPAT_WHEELS%" == "True" (
    for %%f in (%HPAT_WHEELS_DIR%\*) do set hpat_wheel=%%f
	)
if errorlevel 1 exit 1
IF "%HPAT_WHEELS%" == "True" (
    conda remove -y hpat && pip install %hpat_wheel% && python -c "import hpat"
    )
if errorlevel 1 exit 1

@rem TODO investigate root cause of NumbaPerformanceWarning
@rem http://numba.pydata.org/numba-doc/latest/user/parallel.html#diagnostics
IF "%HPAT_NUM_PES%" == "" (
    python -W ignore -u -m hpat.runtests -v
    ) ELSE (
    mpiexec -localonly -n %HPAT_NUM_PES% python -W ignore -u -m hpat.runtests -v)
if errorlevel 1 exit 1

REM Link check for Documentation using Sphinx's in-built linkchecker
REM sphinx-build -b linkcheck -j1 usersource _build/html
REM if errorlevel 1 exit 1

