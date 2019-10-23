set -ex

HDF5_DIR="${PREFIX}" MACOSX_DEPLOYMENT_TARGET=10.9 \
$PYTHON setup.py build install --single-version-externally-managed --record=record.txt

if [ "$HPAT_WHEELS" == "True" ]; then
  if [ -z "$HPAT_WHEELS_DIR" ]; then
    echo "Please set HPAT_WHEELS_DIR to build HPAT wheels"
  else
    # Build HPAT wheel
    echo Build HPAT wheel
    $PYTHON setup.py bdist_wheel
    cp dist/hpat*.whl "$HPAT_WHEELS_DIR"
  fi
fi
