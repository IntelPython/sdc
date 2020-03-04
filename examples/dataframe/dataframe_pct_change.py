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
from numba import njit


@njit
def dataframe_pct_change():
    df = pd.DataFrame({"A": [14, 4, 5, 4, 1, 55],
                       "B": [5, 2, 54, 3, 2, 32],
                       "C": [20, 20, 7, 21, 8, 5],
                       "D": [14, 3, 6, 2, 6, 4]})
    out_df = df.pct_change()

    return out_df

# result DataFrame
#            A          B         C         D
# 0        NaN        NaN       NaN       NaN
# 1  -0.714286  -0.600000  0.000000 -0.785714
# 2   0.250000  26.000000 -0.650000  1.000000
# 3  -0.200000  -0.944444  2.000000 -0.666667
# 4  -0.750000  -0.333333 -0.619048  2.000000
# 5  54.000000  15.000000 -0.375000 -0.333333


print(dataframe_pct_change())
