#!/bin/bash

source activate $CONDA_ENV

find / -name clang-format
ls /usr/lib/ | grep clang
ls /usr/bin/ | grep clang-format

python ./setup.py style

flake8 ./