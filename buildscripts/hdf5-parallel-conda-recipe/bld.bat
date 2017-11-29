
mkdir build
cd build

:: Set environment variables.
set HDF5_EXT_ZLIB=zlib.lib

:: Configure step.
cmake -G "NMake Makefiles" ^
      -D CMAKE_BUILD_TYPE:STRING=RELEASE ^
      -D CMAKE_PREFIX_PATH=%LIBRARY_PREFIX% ^
      -D CMAKE_INSTALL_PREFIX:PATH=%LIBRARY_PREFIX% ^
      -D BUILD_SHARED_LIBS:BOOL=ON ^
      -D HDF5_BUILD_HL_LIB=ON ^
      -D HDF5_BUILD_TOOLS:BOOL=ON ^
      -D ALLOW_UNSUPPORTED=ON ^
      -D HDF5_ENABLE_PARALLEL:BOOL=ON ^
      %SRC_DIR%
if errorlevel 1 exit 1

:: Build C libraries and tools.
nmake
if errorlevel 1 exit 1

:: Install step.
nmake install
if errorlevel 1 exit 1
