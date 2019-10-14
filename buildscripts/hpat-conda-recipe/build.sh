# LDSHARED="mpicxx -cxx=$GXX -shared" LD="mpicxx -cxx=$GXX" \
# CC="mpicxx -cxx=$GXX -std=c++11" GXX="mpicxx -cxx=$GXX -std=c++11" \
# OPENCV_DIR="${PREFIX}" DAALROOT="${PREFIX}"
HDF5_DIR="${PREFIX}" MACOSX_DEPLOYMENT_TARGET=10.9 \
$PYTHON setup.py build install --single-version-externally-managed --record=record.txt

#Build Documentation
#$PYTHON setup.py build_doc
#$PYTHON setup.py build_devdoc
