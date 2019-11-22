# *****************************************************************************
# Copyright (c) 2019, Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************


from ._version import get_versions
import numba

# re-export from Numba
from numba import (typeof, prange, pndindex, gdb, gdb_breakpoint, gdb_init,
                   stencil, threading_layer, jitclass, objmode)
from numba.types import *

import sdc.dict_ext
import sdc.set_ext
from sdc.set_ext import init_set_string
# legacy for STAC A3, TODO: remove
from sdc.dict_ext import (DictIntInt, DictInt32Int32, dict_int_int_type,
                           dict_int32_int32_type)
from sdc.str_ext import string_type
from sdc.str_arr_ext import string_array_type
from numba.types import List
import sdc.compiler
import sdc.io
import sdc.io.np_io
import sdc.hiframes.pd_timestamp_ext
import sdc.hiframes.boxing
import sdc.config
import sdc.timsort
from sdc.decorators import jit

if sdc.config._has_xenon:
    from sdc.io.xenon_ext import read_xenon, xe_connect, xe_open, xe_close

multithread_mode = False


__version__ = get_versions()['version']
del get_versions


if not sdc.config.config_pipeline_hpat_default:
    """
    Overload Numba function to allow call SDC pass in Numba compiler pipeline
    Functions are:
    - Numba DefaultPassBuilder define_nopython_pipeline()

    TODO: Needs to detect 'import Pandas' and align initialization according to it
    """

    sdc.config.numba_compiler_define_nopython_pipeline_orig = numba.compiler.DefaultPassBuilder.define_nopython_pipeline
    numba.compiler.DefaultPassBuilder.define_nopython_pipeline = sdc.datatypes.hpat_pandas_dataframe_pass.sdc_nopython_pipeline_lite_register

def _init_extension():
    '''Register Pandas classes and functions with Numba.

    This exntry_point is called by Numba when it initializes.
    '''
    # Importing SDC is already happened
    pass
