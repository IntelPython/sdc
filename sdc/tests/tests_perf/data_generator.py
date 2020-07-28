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


def gen_arr_from_input(data_length, input_data, random=True, repeat=True, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if random:
        return np.random.choice(input_data, data_length, replace=repeat)
    else:
        return np.asarray(multiply_oneds_data(input_data, data_length))


def gen_arr_of_dtype(data_length, dtype='float', random=True, limits=None, nunique=1000, input_data=None, seed=None):
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
        return gen_arr_from_input(data_length, input_data, random=random)

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


def gen_unique_values(data_length, dtype='int', seed=None):
    """
    data_length: result length of array of unique values,
    dtype: dtype of generated array,
    seed: seed to initialize random state
    """

    if dtype in ('float', 'int'):
        values = np.arange(data_length, dtype=dtype)
    if dtype == 'str':
        values = gen_strlist(data_length)

    return gen_arr_from_input(data_length, values, repeat=False, seed=seed)


def gen_series(data_length,
               dtype='float',
               random=True,
               limits=None,
               nunique=1000,
               input_data=None,
               index_gen=None,
               seed=None):
    """
    data_length: result series length,
    dtype: dtype of generated series,
    limits: a tuple of (min, max) limits for numeric series,
    nunique: number of unique values in generated series,
    input_data: 1D sequence of values used for generation of series data,
    index_gen: callable that will generate index of needed size,
    seed: seed to initialize random state
    """

    if seed is not None:
        np.random.seed(seed)

    # prefer generation based on input data if it's provided
    if input_data is not None:
        series_data = gen_arr_from_input(data_length, input_data, random=random)
    else:
        series_data = gen_arr_of_dtype(data_length, dtype=dtype, limits=limits, nunique=nunique)

    index_data = index_gen(data_length) if index_gen is not None else None
    return pd.Series(series_data, index=index_data)


def gen_df(data_length,
           columns=3,
           col_names=None,
           dtype='float',
           random=True,
           limits=None,
           nunique=1000,
           input_data=None,
           index_gen=None,
           seed=None):
    """
    data_length: result series length,
    dtype: dtype of generated series,
    limits: a tuple of (min, max) limits for numeric series,
    nunique: number of unique values in generated series,
    input_data: 2D sequence of values used for generation of dataframe columns,
    index_gen: callable that will generate index of needed size,
    seed: seed to initialize random state
    """

    if seed is not None:
        np.random.seed(seed)

    col_names = string.ascii_uppercase if col_names is None else col_names
    all_data = []
    for i in range(columns):
        # prefer generation based on input data if it's provided
        if (input_data is not None and i < len(input_data)):
            col_data = gen_arr_from_input(data_length, input_data[i], random=random)
        else:
            col_data = gen_arr_of_dtype(data_length, dtype=dtype, limits=limits, nunique=nunique)
        all_data.append(col_data)

    index_data = index_gen(data_length) if index_gen is not None else None
    return pd.DataFrame(dict(zip(col_names, all_data)), index=index_data)
