CC="mpicc -cc=$CC" CXX="mpicc -cxx=$CXX" ./configure --prefix="${PREFIX}" --enable-parallel
make
make install
