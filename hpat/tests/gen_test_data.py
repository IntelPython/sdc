import h5py
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


class SparkGenerator:

    @staticmethod
    def generate():
        from pyspark.sql import SparkSession
        from pyspark.sql.types import (
            StructType, StructField, DateType, TimestampType)

        # test datetime64, spark dates
        dt1 = pd.DatetimeIndex(['2017-03-03 03:23',
                                '1990-10-23', '1993-07-02 10:33:01'])
        df = pd.DataFrame({'DT64': dt1, 'DATE': dt1.copy()})
        df.to_parquet('pandas_dt.pq')

        spark = SparkSession.builder.appName("GenSparkData").getOrCreate()
        schema = StructType([StructField('DT64', DateType(), True),
                             StructField('DATE', TimestampType(), True)])
        sdf = spark.createDataFrame(df, schema)
        sdf.write.parquet('sdf_dt.pq', 'overwrite')

        spark.stop()


def gen_lr(file_name, N, D):
    points = np.random.random((N, D))
    responses = np.random.random(N)
    f = h5py.File(file_name, "w")
    dset1 = f.create_dataset("points", (N, D), dtype='f8')
    dset1[:] = points
    dset2 = f.create_dataset("responses", (N,), dtype='f8')
    dset2[:] = responses
    f.close()


def generate_other_data():
    N = 101
    D = 10
    gen_lr("lr.hdf5", N, D)

    arr = np.arange(N)
    f = h5py.File("test_group_read.hdf5", "w")
    g1 = f.create_group("G")
    dset1 = g1.create_dataset("data", (N,), dtype='i8')
    dset1[:] = arr
    f.close()

    df = pd.DataFrame({'A': ['bc']+["a"]*3+ ["bc"]*3+['a'], 'B': [-8,1,2,3,1,5,6,7]})
    df.to_parquet("groupby3.pq")

    df = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",
                            "bar", "bar", "bar", "bar"],
                        "B": ["one", "one", "one", "two", "two",
                            "one", "one", "two", "two"],
                        "C": ["small", "large", "large", "small",
                            "small", "large", "small", "small",
                            "large"],
                        "D": [1, 2, 2, 6, 3, 4, 5, 6, 9]})
    df.to_parquet("pivot2.pq")

    # CSV reader test
    data = ("0,2.3,4.6,47736\n"
            "1,2.3,4.6,47736\n"
            "2,2.3,4.6,47736\n"
            "4,2.3,4.6,47736\n")

    with open("csv_data1.csv", "w") as f:
        f.write(data)

    with open("csv_data_infer1.csv", "w") as f:
        f.write('A,B,C,D\n'+data)

    data = ("0,2.3,2015-01-03,47736\n"
            "1,2.3,1966-11-13,47736\n"
            "2,2.3,1998-05-21,47736\n"
            "4,2.3,2018-07-11,47736\n")

    with open("csv_data_date1.csv", "w") as f:
        f.write(data)

    # generated data for parallel merge_asof testing
    df1 = pd.DataFrame({'time': pd.DatetimeIndex(
        ['2017-01-03', '2017-01-06', '2017-02-15', '2017-02-21']),
        'B': [4, 5, 9, 6]})
    df2 = pd.DataFrame({'time': pd.DatetimeIndex(
        ['2017-01-01', '2017-01-14', '2017-01-16', '2017-02-23', '2017-02-23',
        '2017-02-25']), 'A': [2,3,7,8,9,10]})
    df1.to_parquet("asof1.pq")
    df2.to_parquet("asof2.pq")


if __name__ == "__main__":
    print('generation phase')
    ParquetGenerator.gen_kde_pq()
    ParquetGenerator.gen_pq_test()
    SparkGenerator.generate()
    generate_other_data()
