# *****************************************************************************
# Copyright (c) 2020, Intel Corporation All rights reserved.
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


from __future__ import print_function, division, absolute_import
import warnings
from collections import namedtuple
import itertools

import numba
from numba import types
from numba.core import ir, ir_utils
from numba.core import compiler as numba_compiler
from numba.core.registry import CPUDispatcher

from numba.core.ir_utils import (mk_unique_var, replace_vars_inner, find_topo_order,
                            dprint_func_ir, remove_dead, mk_alloc, remove_dels,
                            get_name_var_table, replace_var_names,
                            add_offset_to_labels, get_ir_of_code, find_const,
                            compile_to_numba_ir, replace_arg_nodes,
                            find_callname, guard, require, get_definition,
                            build_definitions, replace_vars_stmt,
                            replace_vars_inner, find_build_sequence)

from numba.core.inline_closurecall import inline_closure_call
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.compiler_machinery import FunctionPass, register_pass

import sdc
from sdc import config
# from sdc.utilities import utils
import sdc.io
from sdc.io import parquet_pio
from sdc.hiframes import join, aggregate, sort
from sdc.utilities.utils import (get_constant, NOT_CONSTANT, debug_prints,
                                 inline_new_blocks, ReplaceFunc, is_call, is_assign, update_globals)
import sdc.hiframes.api
from sdc.str_ext import string_type
from sdc.str_arr_ext import string_array_type
import sdc.io
from sdc.io import csv_ext

import pandas as pd
import numpy as np
import math
import sdc.io
from sdc.io.parquet_pio import ParquetHandler
from sdc.hiframes.pd_timestamp_ext import (datetime_date_type,
                                            datetime_date_to_int, int_to_datetime_date)
from sdc.hiframes.pd_series_ext import SeriesType
from sdc.hiframes.pd_categorical_ext import PDCategoricalDtype, CategoricalArray
from sdc.hiframes.rolling import get_rolling_setup_args, supported_rolling_funcs
from sdc.hiframes.aggregate import get_agg_func, supported_agg_funcs
import sdc.hiframes.pd_dataframe_ext


def remove_hiframes(rhs, lives, call_list):
    # used in stencil generation of rolling
    if len(call_list) == 1 and call_list[0] in [int, min, max, abs]:
        return True
    # used in stencil generation of rolling
    if (len(call_list) == 1 and isinstance(call_list[0], CPUDispatcher)
            and call_list[0].py_func == numba.stencils.stencilparfor._compute_last_ind):
        return True
    # used in stencil generation of rolling
    if call_list == ['ceil', math]:
        return True
    if (len(call_list) == 4 and call_list[1:] == ['api', 'hiframes', sdc] and
            call_list[0] in ['fix_df_array', 'fix_rolling_array',
                             'concat', 'count', 'mean', 'quantile', 'var',
                             'str_contains_regex', 'str_contains_noregex', 'column_sum',
                             'nunique', 'init_series', 'init_datetime_index',
                             'convert_tup_to_rec', 'convert_rec_to_tup']):
        return True
    if (len(call_list) == 4 and call_list[1:] == ['series_kernels', 'hiframes', sdc] and
            call_list[0]
            in ['_sum_handle_nan', '_mean_handle_nan', '_var_handle_nan']):
        return True
    if call_list == ['dist_return', 'distributed_api', sdc]:
        return True
    if call_list == ['init_dataframe', 'pd_dataframe_ext', 'hiframes', sdc]:
        return True
    if call_list == ['get_dataframe_data', 'pd_dataframe_ext', 'hiframes', sdc]:
        return True
    if call_list == ['get_dataframe_index', 'pd_dataframe_ext', 'hiframes', sdc]:
        return True
    # if call_list == ['set_parent_dummy', 'pd_dataframe_ext', 'hiframes', sdc]:
    #     return True
    if call_list == ['rolling_dummy', 'pd_rolling_ext', 'hiframes', sdc]:
        return True
    if call_list == ['agg_typer', 'api', 'hiframes', sdc]:
        return True
    if call_list == [list]:
        return True
    if call_list == ['groupby']:
        return True
    if call_list == ['rolling']:
        return True
    if call_list == [pd.api.types.CategoricalDtype]:
        return True
    # TODO: move to Numba
    if call_list == ['empty_inferred', 'ndarray', 'unsafe', numba]:
        return True
    if call_list == ['chain', itertools]:
        return True
    return False


numba.core.ir_utils.remove_call_handlers.append(remove_hiframes)


@register_pass(mutates_CFG=True, analysis_only=False)
class HiFramesPass(FunctionPass):
    """analyze and transform hiframes calls"""

    _name = "sdc_extention_hi_frames_pass"

    def __init__(self):
        pass

    def run_pass(self, state):
        return HiFramesPassImpl(state).run_pass()


class HiFramesPassImpl(object):

    def __init__(self, state):
        # replace inst variables as determined previously during the pass
        # currently use to keep lhs of Arg nodes intact
        self.replace_var_dict = {}

        # df_var -> {col1:col1_var ...}
        self.df_vars = {}
        # df_var -> label where it is defined
        self.df_labels = {}

        self.arrow_tables = {}
        self.reverse_copies = {}

        # self.state = state
