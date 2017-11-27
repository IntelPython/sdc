"%PYTHON%" setup.py configure --hdf5="%HDF5_DIR%"
if errorlevel 1 exit 1
"%PYTHON%" setup.py build install --single-version-externally-managed --record=record.txt
if errorlevel 1 exit 1
