CC="mpicc -cc=$CC" CXX="mpicxx -cxx=$CXX" ./configure --prefix="${PREFIX}" --enable-parallel
make
make install
