MACOSX_DEPLOYMENT_TARGET=10.9 $PYTHON setup.py configure --hdf5="${PREFIX}"
MACOSX_DEPLOYMENT_TARGET=10.9 "${PYTHON}" -m pip install . --no-deps --ignore-installed --no-cache-dir -vvv
