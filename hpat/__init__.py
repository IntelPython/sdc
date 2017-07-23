from __future__ import print_function, division, absolute_import

import numba
from numba import *
import hpat.dict_ext
from hpat.dict_ext import DictIntInt, dict_int_int_type

def jit(signature_or_function=None, **options):
    from .compiler import add_hpat_stages
    # set nopython by default
    if 'nopython' not in options:
        options['nopython'] = True
    options['parallel'] = True
    return numba.jit(signature_or_function, user_pipeline_funcs=[add_hpat_stages], **options)
