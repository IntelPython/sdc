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


import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd


class ParquetGenerator:
    GEN_KDE_PQ_CALLED = False
    GEN_PQ_TEST_CALLED = False

    @classmethod
    def gen_kde_pq(cls, file_name='kde.parquet', N=101):
        if not cls.GEN_KDE_PQ_CALLED:
            df = pd.DataFrame({'points': np.random.random(N)})
            table = pa.Table.from_pandas(df)
            row_group_size = 128
            pq.write_table(table, file_name, row_group_size)
            cls.GEN_KDE_PQ_CALLED = True

    @classmethod
    def gen_pq_test(cls):
        if not cls.GEN_PQ_TEST_CALLED:
            df = pd.DataFrame(
                {
                    'one': [-1, np.nan, 2.5, 3., 4., 6., 10.0],
                    'two': ['foo', 'bar', 'baz', 'foo', 'bar', 'baz', 'foo'],
                    'three': [True, False, True, True, True, False, False],
                    # float without NA
                    'four': [-1, 5.1, 2.5, 3., 4., 6., 11.0],
                    # str with NA
                    'five': ['foo', 'bar', 'baz', None, 'bar', 'baz', 'foo'],
                }
            )
            table = pa.Table.from_pandas(df)
            pq.write_table(table, 'example.parquet')
            pq.write_table(table, 'example2.parquet', row_group_size=2)
            cls.GEN_PQ_TEST_CALLED = True


def generate_spark_data():
    # test datetime64, spark dates
    dt1 = pd.DatetimeIndex(['2017-03-03 03:23',
                            '1990-10-23', '1993-07-02 10:33:01'])
    df = pd.DataFrame({'DT64': dt1, 'DATE': dt1.copy()})
    df.to_parquet('pandas_dt.pq')

    import os
    import shutil
    import tarfile

    if os.path.exists('sdf_dt.pq'):
        shutil.rmtree('sdf_dt.pq')

    sdf_dt_archive = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sdf_dt.pq.bz2')
    tar = tarfile.open(sdf_dt_archive, "r:bz2")
    tar.extractall('.')
    tar.close()


if __name__ == "__main__":
    print('generation phase')
    ParquetGenerator.gen_kde_pq()
    ParquetGenerator.gen_pq_test()
    generate_spark_data()
