
mkdir build
cd build

cmake -G "NMake Makefiles" ^
  -D CMAKE_BUILD_TYPE:STRING=RELEASE ^
  -D CMAKE_PREFIX_PATH=%LIBRARY_PREFIX% ^
  -D CMAKE_INSTALL_PREFIX:PATH=%LIBRARY_PREFIX% ^
  -D PQ_PREFIX=%BUILD_PREFIX%\Library ^
  %SRC_DIR%

if errorlevel 1 exit 1

nmake
if errorlevel 1 exit 1

nmake install
if errorlevel 1 exit 1
