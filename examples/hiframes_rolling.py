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


import pandas as pd
import numpy as np
import sdc


@sdc.jit
def rolling_df1(n):
    df = pd.DataFrame({'A': np.arange(n), 'B': np.random.ranf(n)})
    Ac = df.A.rolling(5).sum()
    return Ac.sum()


@sdc.jit
def rolling_df2(n):
    df = pd.DataFrame({'A': np.arange(n), 'B': np.random.ranf(n)})
    df['moving average'] = df.A.rolling(window=5, center=True).mean()
    return df['moving average'].sum()


@sdc.jit
def rolling_df3(n):
    df = pd.DataFrame({'A': np.arange(n), 'B': np.random.ranf(n)})
    Ac = df.A.rolling(3, center=True).apply(lambda a: a[0] + 2 * a[1] + a[2])
    return Ac.sum()


n = 10
print("sum left window:")
print(rolling_df1(n))
print("mean center window:")
print(rolling_df2(n))
print("custom center window:")
print(rolling_df3(n))
