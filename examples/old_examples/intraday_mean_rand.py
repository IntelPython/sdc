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
import numpy as np
import time
import sdc
from sdc import prange

# adopted from:
# http://www.pythonforfinance.net/2017/02/20/intraday-stock-mean-reversion-trading-backtest-in-python/


@sdc.jit
def intraday_mean_revert():
    nsyms = 1000
    max_num_days = 80000
    all_res = 0.0

    t1 = time.time()
    for i in prange(nsyms):
        # np.random.seed(0)
        s_open = 20 * np.random.randn(max_num_days)
        s_low = 18 * np.random.randn(max_num_days)
        s_close = 19 * np.random.randn(max_num_days)
        df = pd.DataFrame({'Open': s_open, 'Low': s_low,
                           'Close': s_close})

        # create column to hold our 90 day rolling standard deviation
        df['Stdev'] = df['Close'].rolling(window=90).std()

        # create a column to hold our 20 day moving average
        df['Moving Average'] = df['Close'].rolling(window=20).mean()

        # create a column which holds a TRUE value if the gap down from previous day's low to next
        # day's open is larger than the 90 day rolling standard deviation
        df['Criteria1'] = (df['Open'] - df['Low'].shift(1)) < -df['Stdev']

        # create a column which holds a TRUE value if the opening price of the stock is above the 20 day moving average
        df['Criteria2'] = df['Open'] > df['Moving Average']

        # create a column that holds a TRUE value if both above criteria are also TRUE
        df['BUY'] = df['Criteria1'] & df['Criteria2']

        # calculate daily % return series for stock
        df['Pct Change'] = (df['Close'] - df['Open']) / df['Open']

        # create a strategy return series by using the daily stock returns where the trade criteria above are met
        df['Rets'] = df['Pct Change'][df['BUY']]

        all_res += df['Rets'].mean()

    print(all_res)
    print("execution time:", time.time() - t1)


intraday_mean_revert()
