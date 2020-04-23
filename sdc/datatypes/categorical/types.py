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

"""
Numba types for support pandas Categorical.
"""

from numba import types

import numpy as np


__all__ = [
    'CategoricalDtypeType',
    'Categorical',
]


# TODO: consider renaming to CategoricalDtype b/c Categorical - not CategoricalType
class CategoricalDtypeType(types.Opaque):
    """
    Numba type for pandas CategoricalDtype.

    Contains:
        categories -> array-like
        ordered -> bool
    """
    def __init__(self, categories=None, ordered=None):
        self.categories = categories
        self.ordered = ordered
        name = 'CategoricalDtype(categories={}, ordered={})'.format(
            self.categories, self.ordered)
        super().__init__(name=name)

    def __len__(self):
        return len(self.categories) if self.categories else 0

    @property
    def dtype(self):
        # TODO: take dtype from categories array
        return types.int64

    def int_type(self):
        """
        Return minimal int type to fit all categories.
        """
        dtype = types.int64
        n_cats = len(self.categories)
        if n_cats < np.iinfo(np.int8).max:
            dtype = types.int8
        elif n_cats < np.iinfo(np.int16).max:
            dtype = types.int16
        elif n_cats < np.iinfo(np.int32).max:
            dtype = types.int32
        return dtype


# TODO: make ArrayCompatible. It will make reuse Array boxing, unboxing.
class Categorical(types.Type):
    """
    Numba type for pandas Categorical.

    Contains:
        codes -> array-like
        dtype -> CategoricalDtypeType
    """
    def __init__(self, dtype, codes=None):
        assert(isinstance(dtype, CategoricalDtypeType))
        self.pd_dtype = dtype
        self.codes = codes or types.Array(self.pd_dtype.int_type(), ndim=1, layout='C')
        # TODO: store dtype for categories values and use it for dtype
        super().__init__(name=self.__repr__())

    def __repr__(self):
        def Array__repr__(array):
            return "Array({}, {}, {})".format(
                self.codes.dtype.__repr__(),
                self.codes.ndim.__repr__(),
                self.codes.layout.__repr__()
            )

        dtype = self.pd_dtype.__repr__()
        codes = Array__repr__(self.codes)
        return 'Categorical(dtype={}, codes={})'.format(dtype, codes)

    @property
    def categories(self):
        return self.pd_dtype.categories

    # Properties for model

    @property
    def ndim(self):
        return self.codes.ndim

    @property
    def dtype(self):
        return self.codes.dtype
