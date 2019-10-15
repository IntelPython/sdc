#!/bin/bash

source activate $CONDA_ENV

python -m hpat.tests.gen_test_data

export PYTHONPATH=.
coverage erase
coverage run --source=./hpat --omit ./hpat/ml/*,./hpat/xenon_ext.py,./hpat/ros.py,./hpat/cv_ext.py,./hpat/tests/* -m hpat.runtests