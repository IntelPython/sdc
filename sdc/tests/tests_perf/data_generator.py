import numpy as np
import pandas as pd
import string

from sdc.tests.tests_perf.test_perf_utils import *
from sdc.tests.test_series import gen_strlist


def gen_str_data(data_num, data_length, input_data, data_width):
    data = []
    full_input_data_length = data_width

    data.append(perf_data_gen_fixed_len(input_data, full_input_data_length, data_length))
    for i in range(data_num - 1):
        np.random.seed(i)
        data.append(np.random.choice(input_data, data_length))

    return data


def gen_series_fixed_str(data_num, data_length, input_data, data_width):
    all_data = gen_str_data(data_num, data_length, input_data, data_width)
    results = []
    for data in all_data:
        test_data = pd.Series(data)
        results.append(test_data)

    return results


def gen_arr_of_dtype(data_length, dtype='float', limits=None, nunique=1000, input_data=None, seed=None):
    """
    data_length: result array length,
    dtype: dtype of generated array,
    limits: a tuple of (min, max) limits for numeric arrays,
    nunique: number of unique values in generated array,
    input_data: 1D sequence of values used for generation of array data,
    seed: seed to initialize random state
    """

    if seed is not None:
        np.random.seed(seed)

    # prefer generation based on input data if it's provided
    if input_data is not None:
        return np.random.choice(input_data, data_length)

    if dtype == 'float':
        return np.random.ranf(data_length)
    if dtype == 'int':
        default_limits = (np.iinfo(dtype).min, np.iinfo(dtype).max)
        min_value, max_value = limits or default_limits
        return np.random.randint(min_value, max_value, data_length)
    if dtype == 'str':
        default_strings = gen_strlist(nunique)
        return np.random.choice(default_strings, data_length)
    if dtype == 'bool':
        return np.random.choice([True, False], data_length)

    return None


def gen_series(data_length, dtype='float', limits=None, nunique=1000, input_data=None, seed=None):
    """
    data_length: result series length,
    dtype: dtype of generated series,
    limits: a tuple of (min, max) limits for numeric series,
    nunique: number of unique values in generated series,
    input_data: 1D sequence of values used for generation of series data,
    seed: seed to initialize random state
    """

    if seed is not None:
        np.random.seed(seed)

    # prefer generation based on input data if it's provided
    if input_data is not None:
        series_data = np.random.choice(input_data, data_length)
    else:
        series_data = gen_arr_of_dtype(data_length, dtype=dtype, limits=limits, nunique=nunique)

    # TODO: support index generation
    return pd.Series(series_data)


def gen_df(data_length,
           columns=3,
           col_names=None,
           dtype='float',
           limits=None,
           nunique=1000,
           input_data=None,
           seed=None):
    """
    data_length: result series length,
    dtype: dtype of generated series,
    limits: a tuple of (min, max) limits for numeric series,
    nunique: number of unique values in generated series,
    input_data: 2D sequence of values used for generation of dataframe columns,
    seed: seed to initialize random state
    """

    if seed is not None:
        np.random.seed(seed)

    col_names = string.ascii_uppercase if col_names is None else col_names
    all_data = []
    for i in range(columns):
        # prefer generation based on input data if it's provided
        if (input_data is not None and i < len(input_data)):
            col_data = np.random.choice(input_data[i], data_length)
        else:
            col_data = gen_arr_of_dtype(data_length, dtype=dtype, limits=limits, nunique=nunique)
        all_data.append(col_data)

    # TODO: support index generation
    return pd.DataFrame(dict(zip(col_names, all_data)))
