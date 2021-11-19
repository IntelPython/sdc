# -*- coding: utf-8 -*-
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

import string
import unittest

import numba
import numpy as np
import pandas
from itertools import islice, permutations
from numba.core.errors import TypingError

import sdc
from sdc.config import config_inline_overloads, config_use_parallel_overloads
from sdc.utilities.sdc_typing_utils import TypeChecker

test_global_input_data_unicode_kind4 = [
    '¡Y tú quién te crees?',
    '🐍⚡',
    '大处 着眼，c小处着手c。大大c大处',
]

test_global_input_data_unicode_kind1 = [
    'ascii',
    '12345',
    '1234567890',
]

min_float64 = np.finfo('float64').min
max_float64 = np.finfo('float64').max

test_global_input_data_float64 = [
    [1., -1., 0.1, min_float64, max_float64, max_float64, min_float64, -0.1],
    [1., np.nan, -1., 0., min_float64, max_float64, max_float64, min_float64],
    [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
    [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO],
]


def gen_srand_array(size, nchars=8):
    """Generate array of strings of specified size based on [a-zA-Z] + [0-9]"""
    accepted_chars = list(string.ascii_letters + string.digits)
    rands_chars = np.array(accepted_chars, dtype=(np.str_, 1))

    np.random.seed(100)
    return np.random.choice(rands_chars, size=nchars * size).view((np.str_, nchars))


def gen_frand_array(size, min=-100, max=100, nancount=0):
    """Generate array of float of specified size based on [-100-100]"""
    np.random.seed(100)
    res = (max - min) * np.random.sample(size) + min
    if nancount:
        res[np.random.choice(np.arange(size), nancount)] = np.nan
    return res


def gen_strlist(size, nchars=8, accepted_chars=None):
    """Generate list of strings of specified size based on accepted_chars"""
    if not accepted_chars:
        accepted_chars = string.ascii_letters + string.digits
    generated_chars = islice(permutations(accepted_chars, nchars), size)

    return [''.join(chars) for chars in generated_chars]


def gen_int_df_index(length):
    """Generate random integer index for DataFrame"""
    arr = np.arange(length)
    np.random.seed(0)
    np.random.shuffle(arr)

    return arr


def gen_df(input_data, with_index=False):
    """Generate DataFrame based on list of data like a [[1, 2, 3], [4, 5, 6]]"""
    length = min(len(d) for d in input_data)
    data = {n: d[:length] for n, d in zip(string.ascii_uppercase, input_data)}

    index = None
    if with_index:
        index = gen_int_df_index(length)

    return pandas.DataFrame(data, index=index)


def gen_df_int_cols(input_data, with_index=False):
    """Generate DataFrame based on list of data like a [[1, 2, 3], [4, 5, 6]]"""
    length = min(len(d) for d in input_data)
    data = {n: d[:length] for n, d in enumerate(input_data)}

    index = None
    if with_index:
        index = gen_int_df_index(length)

    return pandas.DataFrame(data, index=index)


def count_array_REPs():
    # from sdc.distributed import Distribution
    # vals = sdc.distributed.dist_analysis.array_dists.values()
    # return sum([v == Distribution.REP for v in vals])
    return 0


def count_parfor_REPs():
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
    wrapper = unittest.skip(msg or "numba pipeline not supported")
    if sdc.config.test_expected_failure:
        wrapper = unittest.expectedFailure
    # wrapper = lambda f: f  # disable skipping
    return wrapper(func) if func else wrapper


def sdc_limitation(func):
    return unittest.expectedFailure(func)


def skip_parallel(msg_or_func):
    msg, func = msg_and_func(msg_or_func)
    wrapper = unittest.skipIf(config_use_parallel_overloads, msg or "fails in parallel mode")
    if sdc.config.test_expected_failure:
        wrapper = unittest.expectedFailure
    # wrapper = lambda f: f  # disable skipping
    return wrapper(func) if func else wrapper


def skip_inline(msg_or_func):
    msg, func = msg_and_func(msg_or_func)
    wrapper = unittest.skipIf(config_inline_overloads, msg or "fails in inline mode")
    if sdc.config.test_expected_failure:
        wrapper = unittest.expectedFailure
    # wrapper = lambda f: f  # disable skipping
    return wrapper(func) if func else wrapper


def take_k_elements(k, data, repeat=False, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.choice(np.asarray(data), k, replace=repeat)


def create_series_from_values(size, data_values, index_values=None, name=None, unique=True, seed=None):
    if seed is not None:
        np.random.seed(seed)

    min_size = min(size, len(data_values))
    if index_values:
        min_size = min(min_size, len(index_values))
    repeat = False if unique and min_size == size else True

    series_data = take_k_elements(size, data_values, repeat)
    series_index = take_k_elements(size, index_values, repeat) if index_values else None

    return pandas.Series(series_data, series_index, name)


def assert_raises_ty_checker(self, err_details, func, *args, **kwargs):
    mapping = {'\n': r'\n\s*', '(': r'\(', ')': r'\)'}
    translation_dict = {ord(k): v for k, v in mapping.items()}
    regex_str = TypeChecker.msg_template.format(*err_details)
    regex_str = regex_str.translate(translation_dict)
    self.assertRaisesRegex(TypingError, regex_str, func, *args, **kwargs)


def assert_pandas_exception(self, test_msg, sdc_exc_str, test_impl, sdc_func, args):
    with self.subTest(test_msg):
        with self.assertRaises(Exception) as context:
            test_impl(*args)
        pandas_exception = context.exception

        with self.assertRaises(type(pandas_exception)) as context:
            sdc_func(*args)
        sdc_exception = context.exception
        self.assertIsInstance(sdc_exception, type(pandas_exception))
        self.assertIn(sdc_exc_str, str(sdc_exception))


def _make_func_from_text(func_text, func_name='test_impl', global_vars={}):
    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    test_impl = loc_vars[func_name]
    return test_impl


def assert_nbtype_for_varname(self, disp, var, expected_type, fn_sig=None):
    fn_sig = fn_sig or disp.nopython_signatures[0]
    cres = disp.get_compile_result(fn_sig)
    fn_typemap = cres.type_annotation.typemap
    self.assertIsInstance(fn_typemap[var], expected_type)
