# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2019-2020, Intel Corporation All rights reserved.
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
import numpy as np
import operator
import pandas as pd

from numba import types
from numba.core import cgutils
from numba.extending import (typeof_impl, NativeValue, intrinsic, box, unbox, lower_builtin, )
from numba.core.errors import TypingError
from numba.core.typing.templates import signature
from numba.core.imputils import impl_ret_untracked, call_getiter

from sdc.datatypes.common_functions import SDCLimitation, _sdc_take
from sdc.utilities.utils import sdc_overload, sdc_overload_attribute, sdc_overload_method
from sdc.utilities.sdc_typing_utils import TypeChecker, check_is_numeric_array, check_signed_integer
from sdc.functions.numpy_like import getitem_by_mask
from sdc.functions.numpy_like import astype as nplike_astype
from numba.core.boxing import box_array, unbox_array


def _check_dtype_param_type(dtype):
    """ Returns True is dtype is a valid type for dtype parameter and False otherwise.
        Used in RangeIndex ctor and other methods that take dtype parameter. """

    valid_dtype_types = (types.NoneType, types.Omitted, types.UnicodeType, types.NumberClass)
    return isinstance(dtype, valid_dtype_types) or dtype is None


