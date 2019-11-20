import sdc
import pandas


def pandas_read_csv():
    """
    pandas.read_csv()
    """
    df = pandas.read_csv('/localdisk/spokhode/sdc/data_100000.csv')
    return df


def test_pandas_read_csv(benchmark):
    result = benchmark(pandas_read_csv)

    # no need to compare with itself
    # pandas.testing.assert_frame_equal(result, pandas_read_csv())


def test_sdc_pandas_read_csv(benchmark):
    # use old implementation via Pandas
    sdc.io.csv_ext._gen_csv_reader_py = sdc.io.csv_ext._gen_csv_reader_py_pandas

    jitted = sdc.jit(pandas_read_csv)
    result = benchmark(jitted)

    pandas.testing.assert_frame_equal(result, pandas_read_csv())


def test_sdc_pyarrow_read_csv(benchmark):
    # use new implementation via PyArrow
    sdc.io.csv_ext._gen_csv_reader_py = sdc.io.csv_ext._gen_csv_reader_py_pyarrow

    jitted = sdc.jit(pandas_read_csv)
    result = benchmark(jitted)

    pandas.testing.assert_frame_equal(result, pandas_read_csv())
