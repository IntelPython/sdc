"%PYTHON%" setup.py configure --hdf5="%LIBRARY_PREFIX%" --hdf5-version=%hdf5%
if errorlevel 1 exit 1

"%PYTHON%" -m pip install . --no-deps --ignore-installed --no-cache-dir -vvv
if errorlevel 1 exit 1
