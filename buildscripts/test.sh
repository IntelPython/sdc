#!/bin/bash

python -m unittest
mpiexec -n 2 python -m unittest
mpiexec -n 3 python -m unittest
