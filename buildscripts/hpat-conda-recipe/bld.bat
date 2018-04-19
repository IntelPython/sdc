set HDF5_DIR="%LIBRARY_PREFIX%"
set OPENCV_DIR="%LIBRARY_PREFIX%"
REM set DAALROOT="%LIBRARY_PREFIX%"
"%PYTHON%" setup.py build install --single-version-externally-managed --record=record.txt
if errorlevel 1 exit 1
