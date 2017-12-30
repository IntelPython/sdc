
mkdir build
cd build

cmake -G "NMake Makefiles" ^
      -D CMAKE_BUILD_TYPE:STRING=RELEASE ^
      -DCMAKE_INSTALL_PREFIX=%LIBRARY_PREFIX% ^
      -DCMAKE_INSTALL_LIBDIR=%LIBRARY_PREFIX%/lib -DPREFIX=%LIBRARY_PREFIX% ..
if errorlevel 1 exit 1

nmake
if errorlevel 1 exit 1

nmake install
if errorlevel 1 exit 1
