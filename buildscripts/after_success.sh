#!/bin/bash

source activate $CONDA_ENV

if [ "$HPAT_RUN_COVERAGE" == "True" ]; then
    coverage combine
    coveralls -v
fi
