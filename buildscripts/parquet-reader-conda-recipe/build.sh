mkdir -p build
pushd build
rm -rf *
cmake -DCMAKE_BUILD_TYPE=release -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCMAKE_INSTALL_LIBDIR=$PREFIX/lib -DPQ_PREFIX=$PREFIX ..
make VERBOSE=1
make install
popd
