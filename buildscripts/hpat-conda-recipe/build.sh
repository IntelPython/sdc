# LDSHARED="mpicxx -cxx=$GXX -shared" LD="mpicxx -cxx=$GXX" \
# CC="mpicxx -cxx=$GXX -std=c++11" GXX="mpicxx -cxx=$GXX -std=c++11" \
HDF5_DIR="${PREFIX}" MACOSX_DEPLOYMENT_TARGET=10.9 \
$PYTHON setup.py build install --single-version-externally-managed --record=record.txt
