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
from pandas_datareader import data
import h5py


def main():
    stocks = pd.read_csv('all_syms.csv')
    file_name = "stock_data_all_google.hdf5"
    f = h5py.File(file_name, "w")

    for symbol in stocks.Symbol:
        try:
            df = data.DataReader(symbol, 'google', start='1/1/2000')
        except BaseException:
            continue
        N = len(df)
        grp = f.create_group(symbol)
        grp.create_dataset("Open", (N,), dtype='f8')[:] = df["Open"]
        grp.create_dataset("High", (N,), dtype='f8')[:] = df["High"]
        grp.create_dataset("Low", (N,), dtype='f8')[:] = df["Low"]
        grp.create_dataset("Close", (N,), dtype='f8')[:] = df["Close"]
        grp.create_dataset("Volume", (N,), dtype='f8')[:] = df["Volume"]
        grp.create_dataset("Date", (N,), dtype='i8')[:] = df.index.values.astype(np.int64)

    f.close()


if __name__ == '__main__':
    main()
