import numba

# re-export from Numba
from numba import (typeof, prange, pndindex, gdb, gdb_breakpoint, gdb_init,
    stencil, threading_layer, jitclass, objmode)
from numba.types import *

import hpat.dict_ext
import hpat.set_ext
from hpat.set_ext import init_set_string
import hpat.distributed_api
from hpat.distributed_api import dist_time
# legacy for STAC A3, TODO: remove
from hpat.dict_ext import (DictIntInt, DictInt32Int32, dict_int_int_type,
    dict_int32_int32_type)
from hpat.str_ext import string_type
from hpat.str_arr_ext import string_array_type
from numba.types import List
from hpat.utils import cprint, distribution_report
import hpat.compiler
import hpat.io
import hpat.io.np_io
import hpat.hiframes.pd_timestamp_ext
import hpat.hiframes.boxing
import hpat.config
import hpat.timsort
from hpat.decorators import jit

if hpat.config._has_xenon:
    from hpat.io.xenon_ext import read_xenon, xe_connect, xe_open, xe_close

multithread_mode = False


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
