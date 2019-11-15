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
import h5py
import time
import sdc
from sdc import prange

# adopted from:
# http://www.pythonforfinance.net/2017/02/20/intraday-stock-mean-reversion-trading-backtest-in-python/


@sdc.jit(locals={'s_open': sdc.float64[:], 's_high': sdc.float64[:],
                  's_low': sdc.float64[:], 's_close': sdc.float64[:],
                  's_vol': sdc.float64[:]})
def intraday_mean_revert():
    file_name = "stock_data_all_google.hdf5"
    f = h5py.File(file_name, "r")
    sym_list = list(f.keys())
    nsyms = len(sym_list)
    max_num_days = 4000
    all_res = np.zeros(max_num_days)

    t1 = time.time()
    for i in prange(nsyms):
        symbol = sym_list[i]

        s_open = f[symbol + '/Open'][:]
        s_high = f[symbol + '/High'][:]
        s_low = f[symbol + '/Low'][:]
        s_close = f[symbol + '/Close'][:]
        s_vol = f[symbol + '/Volume'][:]
        df = pd.DataFrame({'Open': s_open, 'High': s_high, 'Low': s_low,
                           'Close': s_close, 'Volume': s_vol, })

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

        n_days = len(df['Rets'])
        res = np.zeros(max_num_days)
        if n_days:
            res[-n_days:] = df['Rets'].fillna(0).values
        all_res += res

    f.close()
    print(all_res.mean())
    print("execution time:", time.time() - t1)


intraday_mean_revert()
