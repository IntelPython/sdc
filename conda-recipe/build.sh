set -ex

if [ `uname` == Darwin ]; then
    WHEELS_BUILD_ARGS=""
    export MACOSX_DEPLOYMENT_TARGET=10.9
else
    if [ "$CONDA_PY" == "36" ]; then
        WHEELS_BUILD_ARGS="-p manylinux1_x86_64"
    else
        WHEELS_BUILD_ARGS="-p manylinux2014_x86_64"
    fi
fi

$PYTHON setup.py build install --single-version-externally-managed --record=record.txt

# Build wheel package
if [ -n "${WHEELS_OUTPUT_FOLDER}" ]; then
    $PYTHON setup.py bdist_wheel ${WHEELS_BUILD_ARGS}
    cp dist/sdc*.whl ${WHEELS_OUTPUT_FOLDER}
fi
