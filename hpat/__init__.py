from __future__ import print_function, division, absolute_import

import numba
from numba import *
import hpat.dict_ext
import hpat.distributed_api
from hpat.distributed_api import dist_time
from hpat.dict_ext import DictIntInt, DictInt32Int32, dict_int_int_type, dict_int32_int32_type
from hpat.str_ext import string_type
from numba.types import List
from hpat.utils import cprint

def jit(signature_or_function=None, **options):
    from .compiler import add_hpat_stages
    # set nopython by default
    if 'nopython' not in options:
        options['nopython'] = True
    options['parallel'] = True
    return numba.jit(signature_or_function, user_pipeline_funcs=[add_hpat_stages], **options)
