import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import pandas as pd
import argparse
import time
import hpat


def gen_kde(N, file_name):
    # np.random.seed(0)
    df = pd.DataFrame({'points': np.random.random(N)})
    table = pa.Table.from_pandas(df)
    row_group_size = 128
    pq.write_table(table, 'kde.parquet', row_group_size)


def main():
    parser = argparse.ArgumentParser(description='Gen KDE.')
    parser.add_argument('--size', dest='size', type=int, default=2000)
    parser.add_argument('--file', dest='file', type=str, default="kde.hdf5")
    args = parser.parse_args()
    N = args.size
    file_name = args.file

    gen_kde(N, file_name)


if __name__ == '__main__':
    main()
