#!/usr/bin/env python

import runpy
import os
import sdc
import numba


if __name__ == "__main__":
    runpy.run_module('sdc.runtests', run_name='__main__')
