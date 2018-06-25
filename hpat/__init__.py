from __future__ import print_function, division, absolute_import

import numba
from numba import *
import hpat.dict_ext
import hpat.set_ext
from hpat.set_ext import init_set_string
import hpat.distributed_api
from hpat.distributed_api import dist_time
from hpat.dict_ext import DictIntInt, DictInt32Int32, dict_int_int_type, dict_int32_int32_type
from hpat.str_ext import string_type
from hpat.str_arr_ext import string_array_type
from numba.types import List
from hpat.utils import cprint, distribution_report
import hpat.compiler
import hpat.io
import hpat.pd_timestamp_ext
import hpat.config

if hpat.config._has_xenon:
    from hpat.xenon_ext import read_xenon, xe_connect, xe_open, xe_close

multithread_mode = False


def jit(signature_or_function=None, **options):
    # set nopython by default
    if 'nopython' not in options:
        options['nopython'] = True
    #options['parallel'] = True
    options['parallel'] = {'comprehension': True,
                           'setitem':       False,  # FIXME: support parallel setitem
                           'reduction':     True,
                           'numpy':         True,
                           'stencil':       True,
                           'fusion':        True,
                           }

    # this is for previous version of pipeline manipulation (numba hpat_req <0.38)
    # from .compiler import add_hpat_stages
    # return numba.jit(signature_or_function, user_pipeline_funcs=[add_hpat_stages], **options)
    return numba.jit(signature_or_function, pipeline_class=hpat.compiler.HPATPipeline, **options)
