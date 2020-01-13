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

from numba.extending import intrinsic, register_model
from sdc.datatypes.hpat_pandas_rolling_types import (
    gen_hpat_pandas_rolling_init, RollingType, RollingTypeModel)


class DataFrameRollingType(RollingType):
    """Type definition for pandas.DataFrame.rolling functions handling."""
    def __init__(self, data, win_type=None, on=None, closed=None):
        super(DataFrameRollingType, self).__init__('DataFrameRollingType',
                                                   data, win_type=win_type,
                                                   on=on, closed=closed)


@register_model(DataFrameRollingType)
class DataFrameRollingTypeModel(RollingTypeModel):
    """Model for DataFrameRollingType type."""
    def __init__(self, dmm, fe_type):
        super(DataFrameRollingTypeModel, self).__init__(dmm, fe_type)


_hpat_pandas_df_rolling_init = intrinsic(gen_hpat_pandas_rolling_init(
    DataFrameRollingType))
