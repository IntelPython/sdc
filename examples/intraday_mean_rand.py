import pandas as pd
import numpy as np
import h5py
import time
import hpat
from hpat import prange

# adopted from:
# http://www.pythonforfinance.net/2017/02/20/intraday-stock-mean-reversion-trading-backtest-in-python/

@hpat.jit
def intraday_mean_revert():
    nsyms = 500
    max_num_days = 4000
    all_res = np.zeros(max_num_days)

    t1 = time.time()
    for i in prange(nsyms):
        s_open = 20 * np.random.randn(max_num_days)
        s_high = 22 * np.random.randn(max_num_days)
        s_low = 18 * np.random.randn(max_num_days)
        s_close = 19 * np.random.randn(max_num_days)
        s_vol = 1000 * np.random.randn(max_num_days)
        df = pd.DataFrame({'Open': s_open, 'High': s_high, 'Low': s_low,
                            'Close': s_close, 'Volume': s_vol,})

        #create column to hold our 90 day rolling standard deviation
        df['Stdev'] = df['Close'].rolling(window=90).std()

        #create a column to hold our 20 day moving average
        df['Moving Average'] = df['Close'].rolling(window=20).mean()

        #create a column which holds a TRUE value if the gap down from previous day's low to next
        #day's open is larger than the 90 day rolling standard deviation
        df['Criteria1'] = (df['Open'] - df['Low'].shift(1)) < -df['Stdev']

        #create a column which holds a TRUE value if the opening price of the stock is above the 20 day moving average
        df['Criteria2'] = df['Open'] > df['Moving Average']

        #create a column that holds a TRUE value if both above criteria are also TRUE
        df['BUY'] = df['Criteria1'] & df['Criteria2']

        #calculate daily % return series for stock
        df['Pct Change'] = (df['Close'] - df['Open']) / df['Open']

        #create a strategy return series by using the daily stock returns where the trade criteria above are met
        df['Rets'] = df['Pct Change'][df['BUY'] == True]

        all_res += df['Rets'].fillna(0)

    print(all_res.mean())
    print("execution time:", time.time()-t1)

intraday_mean_revert()
