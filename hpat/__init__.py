from __future__ import print_function, division, absolute_import

import numba
from numba import *

from .compiler import add_hpat_stages
set_user_pipeline_func(add_hpat_stages)
del add_hpat_stages

def jit(signature_or_function=None, **options):
    # set nopython by default
    if 'nopython' not in options:
        options['nopython'] = True
    return numba.jit(signature_or_function, **options)
