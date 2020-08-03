# *****************************************************************************
# Copyright (c) 2020, Intel Corporation All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *****************************************************************************

import pandas as pd
import numba
from contextlib import redirect_stdout
import sys
import os
import time

import warnings

warnings.simplefilter("ignore")

# New York Stock Exchange dataset
# Load Data from Kaggle platform https://www.kaggle.com/dgawlik/nyse#prices.csv

out = sys.stdout


def numba_jit(*args, **kwargs):
    kwargs.update({'nopython': True, 'parallel': True})
    return numba.jit(*args, **kwargs)


def process_data():
    t_all = time.time()
    df = pd.read_csv('prices.csv')

    res = (df['open'] + df['close']).sum()

    aver_volume = df["volume"].sum() / df["volume"].size

    df['open'].fillna(-1, inplace=True)
    df['close'].fillna(-1, inplace=True)
    df['low'].fillna(-1, inplace=True)
    df['high'].fillna(-1, inplace=True)
    df['volume'].fillna(-1, inplace=True)

    res = df['open'].max(skipna=True)

    abs_series = df['high'].abs()

    res = abs_series.min(skipna=True)

    res = df['low'].floordiv(100000)
    res = df['high'].floordiv(100)
    res = df['volume'].floordiv(100)

    res = df['open'].map(lambda x: x**2)

    res = df['low'].std()

    end_time = time.time() - t_all

    return df, res, end_time


sdc_process_data = numba_jit(process_data)


def main():
    print("Run Pandas...")
    t_start = time.time()
    process_data()
    print("TOTAL Pandas time: ", time.time() - t_start)

    f = open(os.devnull, 'w')

    with redirect_stdout(f):
        t_start = time.time()
        sdc_process_data()  # Warming up
        t_end = time.time() - t_start
    print("SDC WARM_UP time: ", t_end)

    print("Run SDC...")
    t_start = time.time()
    df1, res, end_time = sdc_process_data()
    t_total = time.time() - t_start
    print("NO boxing SDC time: ", end_time)
    print("TOTAL SDC time: ", t_total)


if __name__ == "__main__":
    main()
