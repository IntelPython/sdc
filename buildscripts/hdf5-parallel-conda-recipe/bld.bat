
mkdir build
cd build

:: Set environment variables.
set HDF5_EXT_ZLIB=zlib.lib

set "CXXFLAGS=%CXXFLAGS% -LTCG"

:: Configure step.
cmake -G "Ninja" ^
      -D CMAKE_BUILD_TYPE:STRING=RELEASE ^
      -D CMAKE_PREFIX_PATH=%LIBRARY_PREFIX% ^
      -D CMAKE_INSTALL_PREFIX:PATH=%LIBRARY_PREFIX% ^
      -D CMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON ^
      -D BUILD_SHARED_LIBS:BOOL=ON ^
      -D HDF5_BUILD_HL_LIB=ON ^
      -D HDF5_BUILD_TOOLS:BOOL=ON ^
      -D HDF5_ENABLE_Z_LIB_SUPPORT:BOOL=ON ^
      -D ALLOW_UNSUPPORTED=ON ^
      -D HDF5_ENABLE_PARALLEL:BOOL=ON ^
      %SRC_DIR%
if errorlevel 1 exit 1

:: Build C libraries and tools.
ninja
if errorlevel 1 exit 1

:: Install step.
ninja install
if errorlevel 1 exit 1
