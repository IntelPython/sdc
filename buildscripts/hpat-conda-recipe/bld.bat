set HDF5_DIR="%LIBRARY_PREFIX%"
REM set OPENCV_DIR="%LIBRARY_PREFIX%"
REM set DAALROOT="%LIBRARY_PREFIX%"
"%PYTHON%" setup.py build install --single-version-externally-managed --record=record.txt
if errorlevel 1 exit 1


REM "%PYTHON%" setup.py build_doc
REM if errorlevel 1 exit 1
REM "%PYTHON%" setup.py build_devdoc
REM if errorlevel 1 exit 1
