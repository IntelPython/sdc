echo on

set HDF5_DIR="%LIBRARY_PREFIX%"
"%PYTHON%" setup.py build install --single-version-externally-managed --record=record.txt
if errorlevel 1 exit 1

@rem Build HPAT wheel
echo Build HPAT wheel
IF "%HPAT_WHEELS%" == "True" (
    "%PYTHON%" setup.py bdist_wheel
    if errorlevel 1 exit 1
    copy dist\hpat*.whl "%HPAT_WHEELS_DIR%"
    if errorlevel 1 exit 1
    )
