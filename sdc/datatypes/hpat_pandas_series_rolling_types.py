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

from numba import cgutils, types
from numba.datamodel import StructModel
from numba.extending import (intrinsic, make_attribute_wrapper,
                             models, register_model)
from numba.typing.templates import signature


class SeriesRollingType(types.Type):
    """Type definition for pandas.Series.rolling functions handling."""
    def __init__(self, data, win_type=None, on=None, closed=None):
        self.data = data
        self.win_type = win_type or types.none
        self.on = on or types.none
        self.closed = closed or types.none

        name_tmpl = 'SeriesRollingType({}, win_type={}, on={}, closed={})'
        name = name_tmpl.format(data, self.win_type, self.on, self.closed)
        super(SeriesRollingType, self).__init__(name)


@register_model(SeriesRollingType)
class SeriesRollingTypeModel(StructModel):
    """Model for SeriesRollingType type."""
    def __init__(self, dmm, fe_type):
        members = [
            ('data', fe_type.data),
            # window is able to be offset
            ('window', types.intp),
            ('min_periods', types.intp),
            ('center', types.boolean),
            ('win_type', fe_type.win_type),
            ('on', fe_type.on),
            # axis is able to be unicode type
            ('axis', types.intp),
            ('closed', fe_type.closed),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(SeriesRollingType, 'data', '_data')
make_attribute_wrapper(SeriesRollingType, 'window', '_window')
make_attribute_wrapper(SeriesRollingType, 'min_periods', '_min_periods')
make_attribute_wrapper(SeriesRollingType, 'center', '_center')
make_attribute_wrapper(SeriesRollingType, 'win_type', '_win_type')
make_attribute_wrapper(SeriesRollingType, 'on', '_on')
make_attribute_wrapper(SeriesRollingType, 'axis', '_axis')
make_attribute_wrapper(SeriesRollingType, 'closed', '_closed')


@intrinsic
def _hpat_pandas_series_rolling_init(typingctx, self, window, min_periods=None,
                                     center=False, win_type=None,
                                     on=None, axis=0, closed=None):
    """Internal Numba required function to register SeriesRollingType."""

    ret_typ = SeriesRollingType(self, win_type, on, closed)
    sig = signature(ret_typ, self, window, min_periods,
                    center, win_type, on, axis, closed)

    def _codegen(context, builder, sig, args):
        """Create SeriesRollingTypeModel structure."""
        data, window, min_periods, center, win_type, on, axis, closed = args
        rolling = cgutils.create_struct_proxy(sig.return_type)(context, builder)
        rolling.data = data
        rolling.window = window
        rolling.min_periods = min_periods
        rolling.center = center
        rolling.win_type = win_type
        rolling.on = on
        rolling.axis = axis
        rolling.closed = closed

        if context.enable_nrt:
            context.nrt.incref(builder, self, rolling.data)

        return rolling._getvalue()

    return sig, _codegen
