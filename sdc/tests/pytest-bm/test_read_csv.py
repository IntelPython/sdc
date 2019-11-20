import sdc
import pandas
import pyarrow.csv

import pytest


def pandas_read_csv():
    """
    pandas.read_csv()
    """
    df = pandas.read_csv('data_100000.csv')
    return df


def pyarrow_read_csv():
    """
    pyarrow.csv.read_csv()
    """
    df = pyarrow.csv.read_csv('data_100000.csv').to_pandas()
    return df


def pandas_read_csv_via_pyarrow():
    """
    pandas_read_csv()
    """
    df = sdc.io.csv_ext.pandas_read_csv('data_100000.csv')
    return df


@pytest.mark.benchmark(group="read_csv")
def test_py_pandas_read_csv(benchmark):
    result = benchmark(pandas_read_csv)

    # no need to compare with itself
    # pandas.testing.assert_frame_equal(result, pandas_read_csv())


@pytest.mark.benchmark(group="read_csv")
def test_py_pyarrow_read_csv(benchmark):
    result = benchmark(pyarrow_read_csv)

    pandas.testing.assert_frame_equal(result, pandas_read_csv())


@pytest.mark.benchmark(group="read_csv")
def test_py_pandas_read_csv_via_pyarrow(benchmark):
    result = benchmark(pandas_read_csv_via_pyarrow)

    pandas.testing.assert_frame_equal(result, pandas_read_csv())


@pytest.mark.benchmark(group="read_csv")
def test_sdc_pandas_read_csv(benchmark):
    # use old implementation via Pandas
    sdc.io.csv_ext._gen_csv_reader_py = sdc.io.csv_ext._gen_csv_reader_py_pandas

    jitted = sdc.jit(pandas_read_csv)
    result = benchmark(jitted)

    pandas.testing.assert_frame_equal(result, pandas_read_csv())


@pytest.mark.benchmark(group="read_csv")
def test_sdc_pyarrow_read_csv(benchmark):
    # use new implementation via PyArrow
    sdc.io.csv_ext._gen_csv_reader_py = sdc.io.csv_ext._gen_csv_reader_py_pyarrow

    jitted = sdc.jit(pandas_read_csv)
    result = benchmark(jitted)

    pandas.testing.assert_frame_equal(result, pandas_read_csv())
