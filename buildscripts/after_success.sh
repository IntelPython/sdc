#!/bin/bash

source activate $CONDA_ENV

if [ "$RUN_COVERAGE" == "yes" ]; then
    coverage combine
    coveralls -v
fi
