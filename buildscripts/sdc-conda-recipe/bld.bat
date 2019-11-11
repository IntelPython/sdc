echo on

set HDF5_DIR="%LIBRARY_PREFIX%"
"%PYTHON%" setup.py build install --single-version-externally-managed --record=record.txt
if errorlevel 1 exit 1
