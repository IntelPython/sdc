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

import time
import pandas as pd
from numba import njit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


@njit(parallel=True)
def preprocess_data():
    # Reading stock prices from CSV file
    df_prices = pd.read_csv('./prices.csv')  # Public dataset from https://www.kaggle.com/dgawlik/nyse

    # Select stock of interest (INTC)
    df_prices_intc = df_prices[df_prices['symbol'] == 'INTC']

    # Remove unused columns
    df_prices_intc = df_prices_intc.drop(columns=('symbol', 'volume'))

    # The year of interest is 2012
    df_prices_intc_2012 = df_prices_intc[df_prices_intc['date'] <= '2012-12-31']
    df_prices_intc_2012 = df_prices_intc_2012[df_prices_intc['date'] >= '2012-01-01']

    # Pearson correlation between open and close prices for 2012
    corr_open_close_2012 = df_prices_intc_2012['open'].corr(df_prices_intc_2012['close'])

    # Keep days when started low and finished high in 2012
    df_prices_intc_2012_low2high = df_prices_intc_2012[df_prices_intc['open'] <= df_prices_intc['low']*1.005]
    df_prices_intc_2012_low2high = df_prices_intc_2012_low2high[df_prices_intc['close'] >= df_prices_intc['high']*0.995]

    # Prepare data for forecasting
    x = df_prices_intc_2012['open'].values.reshape(-1, 1)
    y = df_prices_intc_2012['close']

    return df_prices_intc_2012, corr_open_close_2012, df_prices_intc_2012_low2high, x, y


t_start = time.time()
data2012, coc12, data2012_low2high, x, y = preprocess_data()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
regr = GradientBoostingRegressor()
my_fit = regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)
sc = regr.score(x_test, y_test)
t_end = time.time()

print('2012')
print(data2012.head())

print('Pearson correlation')
print(coc12)

print('Days traded low to high')
print(data2012_low2high)

print(y_pred)
print(sc)

print('Execution time: ', t_end - t_start)
