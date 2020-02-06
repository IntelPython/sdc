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


def gen_data(data_num, data_length, input_data):
    data = []
    full_input_data_length = sum(len(i) for i in input_data)
    data.append(perf_data_gen_fixed_len(input_data, full_input_data_length,
                                        data_length))
    for i in range(data_num - 1):
        np.random.seed(i)
        data.append(np.random.ranf(data_length))

    return data


def gen_series_str(data_num, data_length, input_data, data_width):
    datas = gen_str_data(data_num, data_length, input_data, data_width)
    args = []
    for data in datas:
        test_data = pd.Series(data)
        args.append(test_data)

    return args


def gen_series(data_num, data_length, input_data):
    datas = gen_data(data_num, data_length, input_data)
    args = []
    for data in datas:
        test_data = pd.Series(data)
        args.append(test_data)

    return args


def gen_df(data_num, data_length, input_data):
    datas = gen_data(data_num, data_length, input_data)
    args = []
    for data in datas:
        test_data = pandas.DataFrame({f"f{i}": data for i in range(3)})
        args.append(test_data)

    return args


