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


import numba
import sdc

from typing import NamedTuple
from sdc.utilities.utils import sdc_overload, sdc_register_jitable


class Chunk(NamedTuple):
    start: int
    stop: int


@sdc_register_jitable
def get_pool_size():
    if sdc.config.config_use_parallel_overloads:
        return numba.config.NUMBA_NUM_THREADS

    return 1


@sdc_register_jitable
def get_chunks(size, pool_size):
    pool_size = min(pool_size, size)
    chunk_size = size // pool_size
    overload_size = size % pool_size

    chunks = []
    for i in range(pool_size):
        start = i * chunk_size + min(i, overload_size)
        stop = (i + 1) * chunk_size + min(i + 1, overload_size)
        chunks.append(Chunk(start, stop))

    return chunks


@sdc_register_jitable
def parallel_chunks(size):
    return get_chunks(size, get_pool_size())
