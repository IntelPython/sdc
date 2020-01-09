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
from numba.extending import make_attribute_wrapper, models
from numba.typing.templates import signature
from sdc.datatypes.common_functions import TypeChecker


class RollingType(types.Type):
    """Type definition for pandas.rolling functions handling."""
    def __init__(self, ty, data, win_type=None, on=None, closed=None):
        self.data = data
        self.win_type = win_type or types.none
        self.on = on or types.none
        self.closed = closed or types.none

        name_tmpl = '{}({}, win_type={}, on={}, closed={})'
        name = name_tmpl.format(ty, data, self.win_type, self.on, self.closed)
        super(RollingType, self).__init__(name)


class RollingTypeModel(StructModel):
    """Model for RollingType type."""
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


make_attribute_wrapper(RollingType, 'data', '_data')
make_attribute_wrapper(RollingType, 'window', '_window')
make_attribute_wrapper(RollingType, 'min_periods', '_min_periods')
make_attribute_wrapper(RollingType, 'center', '_center')
make_attribute_wrapper(RollingType, 'win_type', '_win_type')
make_attribute_wrapper(RollingType, 'on', '_on')
make_attribute_wrapper(RollingType, 'axis', '_axis')
make_attribute_wrapper(RollingType, 'closed', '_closed')


def gen_hpat_pandas_rolling_init(ty):
    """Generate rolling initializer based on data type"""
    def _hpat_pandas_rolling_init(typingctx, self, window, min_periods=None,
                                  center=False, win_type=None,
                                  on=None, axis=0, closed=None):
        """Internal Numba required function to register RollingType."""
        ret_typ = ty(self, win_type, on, closed)
        sig = signature(ret_typ, self, window, min_periods,
                        center, win_type, on, axis, closed)

        def _codegen(context, builder, sig, args):
            """Create DataFrameRollingTypeModel structure."""
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

    return _hpat_pandas_rolling_init


def gen_sdc_pandas_rolling(initializer, ty):
    """Generate rolling initializer"""
    def sdc_pandas_rolling(self, window, min_periods=None, center=False,
                           win_type=None, on=None, axis=0, closed=None):
        ty_checker = TypeChecker('Method rolling().')
        ty_checker.check(self, ty)

        if not isinstance(window, types.Integer):
            ty_checker.raise_exc(window, 'int', 'window')

        minp_accepted = (types.Omitted, types.NoneType, types.Integer)
        if not isinstance(min_periods, minp_accepted) and min_periods is not None:
            ty_checker.raise_exc(min_periods, 'None, int', 'min_periods')

        center_accepted = (types.Omitted, types.Boolean)
        if not isinstance(center, center_accepted) and center is not False:
            ty_checker.raise_exc(center, 'bool', 'center')

        str_types = (types.Omitted, types.NoneType, types.StringLiteral, types.UnicodeType)
        if not isinstance(win_type, str_types) and win_type is not None:
            ty_checker.raise_exc(win_type, 'str', 'win_type')

        if not isinstance(on, str_types) and on is not None:
            ty_checker.raise_exc(on, 'str', 'on')

        axis_accepted = (types.Omitted, types.Integer, types.StringLiteral, types.UnicodeType)
        if not isinstance(axis, axis_accepted) and axis != 0:
            ty_checker.raise_exc(axis, 'int, str', 'axis')

        if not isinstance(closed, str_types) and closed is not None:
            ty_checker.raise_exc(closed, 'str', 'closed')

        nan_minp = isinstance(min_periods, (types.Omitted, types.NoneType)) or min_periods is None

        def sdc_pandas_rolling_impl(self, window, min_periods=None, center=False,
                                    win_type=None, on=None, axis=0, closed=None):
            if window < 0:
                raise ValueError('window must be non-negative')

            if nan_minp == True:  # noqa
                minp = window
            else:
                minp = min_periods

            if minp < 0:
                raise ValueError('min_periods must be >= 0')
            if minp > window:
                raise ValueError('min_periods must be <= window')

            if center != False:  # noqa
                raise ValueError('Method rolling(). The object center\n expected: False')

            if win_type is not None:
                raise ValueError('Method rolling(). The object win_type\n expected: None')

            if on is not None:
                raise ValueError('Method rolling(). The object on\n expected: None')

            if axis != 0:
                raise ValueError('Method rolling(). The object axis\n expected: 0')

            if closed is not None:
                raise ValueError('Method rolling(). The object closed\n expected: None')

            return initializer(self, window, minp, center, win_type, on, axis, closed)

        return sdc_pandas_rolling_impl

    return sdc_pandas_rolling


sdc_pandas_rolling_docstring_tmpl = """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.{ty}.rolling

    Examples
    --------
    .. literalinclude:: ../../../examples/{ty_lower}/rolling/{ty_lower}_rolling_min.py
       :language: python
       :lines: 27-
       :caption: Calculate the rolling minimum.
       :name: ex_{ty_lower}_rolling

    .. command-output:: python ./{ty_lower}/rolling/{ty_lower}_rolling_min.py
       :cwd: ../../../examples

    .. todo:: Add support of parameters ``center``, ``win_type``, ``on``, ``axis`` and ``closed``

    .. seealso::
        :ref:`expanding <pandas.{ty}.expanding>`
            Provides expanding transformations.
        :ref:`ewm <pandas.{ty}.ewm>`
            Provides exponential weighted functions.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas {ty} attribute :attr:`pandas.{ty}.rolling` implementation
    .. only:: developer

    Test: python -m sdc.runtests -k sdc.tests.test_rolling.TestRolling.test_{ty_lower}_rolling

    Parameters
    ----------
    self: :obj:`pandas.{ty}`
        Input {ty}.
    window: :obj:`int` or :obj:`offset`
        Size of the moving window.
    min_periods: :obj:`int`
        Minimum number of observations in window required to have a value.
    center: :obj:`bool`
        Set the labels at the center of the window.
        *unsupported*
    win_type: :obj:`str`
        Provide a window type.
        *unsupported*
    on: :obj:`str`
        Column on which to calculate the rolling window.
        *unsupported*
    axis: :obj:`int`, :obj:`str`
        Axis along which the operation acts
        0/None/'index' - row-wise operation
        1/'columns'    - column-wise operation
        *unsupported*
    closed: :obj:`str`
        Make the interval closed on the ‘right’, ‘left’, ‘both’ or ‘neither’ endpoints.
        *unsupported*

    Returns
    -------
    :class:`pandas.{ty}.rolling`
        Output class to manipulate with input data.
"""
