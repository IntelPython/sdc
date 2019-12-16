# -*- coding: utf-8 -*-
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

import unittest

import numpy as np

import sdc
import numba


test_global_input_data_unicode_kind4 = [
    '¡Y tú quién te crees?',
    '🐍⚡',
    '大处 着眼，c小处着手c。大大c大处',
    'c小处着手c。',
    'te crees?'
]

min_float64 = np.finfo('float64').min
max_float64 = np.finfo('float64').max
min_int64 = np.iinfo(np.int64).min
max_int64 = np.iinfo(np.int64).max

test_global_input_data_float64 = [
    [1., -1., 0.1, min_float64, max_float64, max_float64, min_float64, -0.1],
    [1., np.nan, -1., 0., min_float64, max_float64, max_float64, min_float64],
    [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
    [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO],
]

test_datatime = np.array(['2007-07-13', '2006-01-13', '2010-08-13',
                         '2005-02-27', '2005-02-28'], dtype='datetime64')

test_global_input_data_int64 = [min_int64, 0, 1, -23, max_int64, min_int64, max_int64]

def count_array_REPs():
    if sdc.config.config_pipeline_hpat_default:
        from sdc.distributed import Distribution
        vals = sdc.distributed.dist_analysis.array_dists.values()
        return sum([v == Distribution.REP for v in vals])
    else:
        return 0


def count_parfor_REPs():
    if sdc.config.config_pipeline_hpat_default:
        from sdc.distributed import Distribution
        vals = sdc.distributed.dist_analysis.parfor_dists.values()
        return sum([v == Distribution.REP for v in vals])
    else:
        return 0


def count_parfor_OneDs():
    from sdc.distributed import Distribution
    vals = sdc.distributed.dist_analysis.parfor_dists.values()
    return sum([v == Distribution.OneD for v in vals])


def count_array_OneDs():
    from sdc.distributed import Distribution
    vals = sdc.distributed.dist_analysis.array_dists.values()
    return sum([v == Distribution.OneD for v in vals])


def count_parfor_OneD_Vars():
    from sdc.distributed import Distribution
    vals = sdc.distributed.dist_analysis.parfor_dists.values()
    return sum([v == Distribution.OneD_Var for v in vals])


def count_array_OneD_Vars():
    from sdc.distributed import Distribution
    vals = sdc.distributed.dist_analysis.array_dists.values()
    return sum([v == Distribution.OneD_Var for v in vals])


def dist_IR_contains(*args):
    return sum([(s in sdc.distributed.fir_text) for s in args])


@sdc.jit
def get_rank():
    return sdc.distributed_api.get_rank()


@sdc.jit
def get_start_end(n):
    rank = sdc.distributed_api.get_rank()
    n_pes = sdc.distributed_api.get_size()
    start = sdc.distributed_api.get_start(n, n_pes, rank)
    end = sdc.distributed_api.get_end(n, n_pes, rank)
    return start, end


def check_numba_version(version):
    return numba.__version__ == version


def msg_and_func(msg_or_func=None):
    if msg_or_func is None:
        # No signature, no function
        func = None
        msg = None
    elif isinstance(msg_or_func, str):
        # A message is passed
        func = None
        msg = msg_or_func
    else:
        # A function is passed
        func = msg_or_func
        msg = None
    return msg, func


def skip_numba_jit(msg_or_func=None):
    msg, func = msg_and_func(msg_or_func)
    wrapper = unittest.skipUnless(sdc.config.config_pipeline_hpat_default, msg or "numba pipeline not supported")
    # wrapper = lambda f: f  # disable skipping
    return wrapper(func) if func else wrapper


def skip_sdc_jit(msg_or_func=None):
    msg, func = msg_and_func(msg_or_func)
    wrapper = unittest.skipIf(sdc.config.config_pipeline_hpat_default, msg or "sdc pipeline not supported")
    # wrapper = lambda f: f  # disable skipping
    return wrapper(func) if func else wrapper
