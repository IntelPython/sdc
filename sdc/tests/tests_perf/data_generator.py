import numpy as np
import pandas as pd

from sdc.tests.tests_perf.test_perf_utils import *


def gen_str_data(data_num, data_length, input_data, data_width):
    data = []
    full_input_data_length = data_width
    data.append(perf_data_gen_fixed_len(input_data, full_input_data_length,
                                        data_length))
    for i in range(data_num - 1):
        np.random.seed(i)
        data.append(np.random.choice(input_data, data_length))

    return data


def gen_data(data_num, data_length, input_data, typ):
    data = []
    full_input_data_length = sum(len(i) for i in input_data)
    data.append(perf_data_gen_fixed_len(input_data, full_input_data_length,
                                        data_length))
    for i in range(data_num - 1):
        np.random.seed(i)
        if typ == 'float':
            data.append(np.random.ranf(data_length))
        elif typ == 'int':
            data.append(np.random.randint(10 ** 4, size=data_length))

    return data


def gen_series_str(data_num, data_length, input_data, data_width):
    all_data = gen_str_data(data_num, data_length, input_data, data_width)
    results = []
    for data in all_data:
        test_data = pd.Series(data)
        results.append(test_data)

    return results


def gen_series(data_num, data_length, input_data, typ):
    all_data = gen_data(data_num, data_length, input_data, typ)
    results = []
    for data in all_data:
        test_data = pd.Series(data)
        results.append(test_data)

    return results


def gen_df(data_num, data_length, input_data, typ, columns=3):
    all_data = gen_data(data_num, data_length, input_data, typ)
    results = []
    for data in all_data:
        test_data = pd.DataFrame({f"f{i}": data for i in range(columns)})
        results.append(test_data)

    return results
