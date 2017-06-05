from __future__ import print_function, division, absolute_import

import numba
from numba import *

def jit(signature_or_function=None, **options):
    from .compiler import add_hpat_stages
    set_user_pipeline_func(add_hpat_stages)
    # set nopython by default
    if 'nopython' not in options:
        options['nopython'] = True
    options['parallel'] = True
    return numba.jit(signature_or_function, **options)
