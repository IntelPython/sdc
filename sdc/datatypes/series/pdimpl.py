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

import pandas as pd

from numba import typeof
from numba import types
from numba import objmode

from ..categorical.pdimpl import _reconstruct_CategoricalDtype


def _reconstruct_Series(data, dtype):
    values_list = [v.literal_value for v in data]
    dtype = _reconstruct_CategoricalDtype(dtype)
    return pd.Series(data=values_list, dtype=dtype)


def _Series_category(data=None, index=None, dtype=None, name=None, copy=False, fastpath=False):
    """
    Implementation of constructor for pandas Series via objmode.
    """
    # TODO: support other parameters (only data and dtype now)

    ty = typeof(_reconstruct_Series(data, dtype))

    from textwrap import dedent
    text = dedent(f"""
    def impl(data=None, index=None, dtype=None, name=None, copy=False, fastpath=False):
        with objmode(series="{ty}"):
            series = pd.Series(data, index, dtype, name, copy, fastpath)
        return series
    """)
    globals, locals = {'objmode': objmode, 'pd': pd}, {}
    exec(text, globals, locals)
    impl = locals['impl']
    return impl
