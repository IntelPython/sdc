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
