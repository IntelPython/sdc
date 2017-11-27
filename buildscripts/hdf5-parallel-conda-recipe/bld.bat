configure --prefix="%PREFIX%" --enable-parallel
if errorlevel 1 exit 1
make
if errorlevel 1 exit 1
make install
if errorlevel 1 exit 1
