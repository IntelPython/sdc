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


import hpat
import numba


test_global_input_data_unicode_kind4 = [
    '¡Y tú quién te crees?',
    '🐍⚡',
    '大处 着眼，c小处着手c。大大c大处',
]


def count_array_REPs():
    from hpat.distributed import Distribution
    vals = hpat.distributed.dist_analysis.array_dists.values()
    return sum([v == Distribution.REP for v in vals])


def count_parfor_REPs():
    from hpat.distributed import Distribution
    vals = hpat.distributed.dist_analysis.parfor_dists.values()
    return sum([v == Distribution.REP for v in vals])


def count_parfor_OneDs():
    from hpat.distributed import Distribution
    vals = hpat.distributed.dist_analysis.parfor_dists.values()
    return sum([v == Distribution.OneD for v in vals])


def count_array_OneDs():
    from hpat.distributed import Distribution
    vals = hpat.distributed.dist_analysis.array_dists.values()
    return sum([v == Distribution.OneD for v in vals])


def count_parfor_OneD_Vars():
    from hpat.distributed import Distribution
    vals = hpat.distributed.dist_analysis.parfor_dists.values()
    return sum([v == Distribution.OneD_Var for v in vals])


def count_array_OneD_Vars():
    from hpat.distributed import Distribution
    vals = hpat.distributed.dist_analysis.array_dists.values()
    return sum([v == Distribution.OneD_Var for v in vals])


def dist_IR_contains(*args):
    return sum([(s in hpat.distributed.fir_text) for s in args])


@hpat.jit
def get_rank():
    return hpat.distributed_api.get_rank()


@hpat.jit
def get_start_end(n):
    rank = hpat.distributed_api.get_rank()
    n_pes = hpat.distributed_api.get_size()
    start = hpat.distributed_api.get_start(n, n_pes, rank)
    end = hpat.distributed_api.get_end(n, n_pes, rank)
    return start, end


def check_numba_version(version):
    return numba.__version__ == version
