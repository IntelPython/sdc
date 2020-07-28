# Copyright 2020 Intel Corporation
# Copyright 2019 NVIDIA Corporation
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# NOTICE: THIS FILE HAS BEEN MODIFIED BY INTEL CORPORATION UNDER COMPLIANCE WITH THE APACHE 2.0 LICENCE FROM
# THE ORIGINAL WORK OF TAUREAN DYER (NVIDIA Corporation)
# https://github.com/rapidsai/notebooks-contrib/blob/branch-0.14/intermediate_notebooks/E2E/census/census_education2income_demo.ipynb

import time
import numba
import numpy as np
import pandas as pd
import sklearn.linear_model as lm

from numba import literal_unroll
from sklearn.model_selection import train_test_split

# Load dataset from:
# https://rapidsai-data.s3.us-east-2.amazonaws.com/datasets/ipums_education2income_1970-2010.csv.gz


def numba_jit(*args, **kwargs):
    kwargs.update({'parallel': True, 'nopython': True})
    return numba.njit(*args, **kwargs)


def pandas_setitem_impl(df, key, value):
    df[key] = value
    return df


@numba_jit
def sdc_setitem_impl(df, key, value):
    return df._set_column(key, value)


def gen_run_etl(setitem_impl=pandas_setitem_impl):
    def run_etl():
        # Read only these columns
        keep_cols = (
            'YEAR', 'DATANUM', 'SERIAL', 'CBSERIAL', 'HHWT', 'GQ', 'PERNUM',
            'SEX', 'AGE', 'INCTOT', 'EDUC', 'EDUCD', 'EDUC_HEAD', 'EDUC_POP',
            'EDUC_MOM', 'EDUCD_MOM2', 'EDUCD_POP2', 'INCTOT_MOM', 'INCTOT_POP',
            'INCTOT_MOM2', 'INCTOT_POP2', 'INCTOT_HEAD', 'SEX_HEAD', 'CPI99'
            )
        dtypes = {
            'YEAR': np.float64,
            'DATANUM': np.float64,
            'SERIAL': np.float64,
            'CBSERIAL': np.float64,
            'HHWT': np.float64,
            'GQ': np.float64,
            'PERNUM': np.float64,
            'SEX': np.float64,
            'AGE': np.float64,
            'INCTOT': np.float64,
            'EDUC': np.float64,
            'EDUCD': np.float64,
            'EDUC_HEAD': np.float64,
            'EDUC_POP': np.float64,
            'EDUC_MOM': np.float64,
            'EDUCD_MOM2': np.float64,
            'EDUCD_POP2': np.float64,
            'INCTOT_MOM': np.float64,
            'INCTOT_POP': np.float64,
            'INCTOT_MOM2': np.float64,
            'INCTOT_POP2': np.float64,
            'INCTOT_HEAD': np.float64,
            'SEX_HEAD': np.float64,
            'CPI99': np.float64
        }
        # https://rapidsai-data.s3.us-east-2.amazonaws.com/datasets/ipums_education2income_1970-2010.csv.gz
        df = pd.read_csv('ipums_education2income_1970-2010.csv',
                         usecols=keep_cols, dtype=dtypes)

        mask = df['INCTOT'] != 9999999

        df = df[mask]
        df = df.reset_index(drop=True)

        res = df['INCTOT'] * df['CPI99']
        df = setitem_impl(df, 'INCTOT', res)

        mask1 = df['EDUC'].notna()

        df = df[mask1]
        df = df.reset_index(drop=True)

        mask2 = df['EDUCD'].notna()

        df = df[mask2]
        df = df.reset_index(drop=True)

        for col in literal_unroll(keep_cols):
            df[col].fillna(-1, inplace=True)

        y = df['EDUC']
        X = df.drop(columns='EDUC')

        return X, y

    return run_etl


run_etl_pandas = gen_run_etl()
run_etl_sdc = numba_jit(gen_run_etl(sdc_setitem_impl))


def mse(y_test, y_pred):
    return ((y_test - y_pred) ** 2).mean()


def cod(y_test, y_pred):
    y_bar = y_test.mean()
    total = ((y_test - y_bar) ** 2).sum()
    residuals = ((y_test - y_pred) ** 2).sum()
    return 1 - (residuals / total)


def train_and_test(X, y): 
    clf = lm.Ridge()

    mse_values, cod_values = [], []
    N_RUNS = 50
    TRAIN_SIZE = 0.9
    random_state = 777

    for i in range(N_RUNS):
        train_test_result = train_test_split(X, y, train_size=TRAIN_SIZE,
                                             random_state=random_state,
                                             shuffle=False)
        X_train, X_test, y_train, y_test = train_test_result
        random_state += 777
        model = clf.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse_values.append(mse(y_test, y_pred))
        cod_values.append(cod(y_test, y_pred))

    return mse_values, cod_values


def run_ml(X, y):
    mse_values, cod_values = train_and_test(X, y)

    mean_mse = sum(mse_values) / len(mse_values)
    mean_cod = sum(cod_values) / len(cod_values)
    mse_dev = pow(
        sum([(mse_value - mean_mse) ** 2 for mse_value in mse_values]) / (
        len(mse_values) - 1), 0.5)
    cod_dev = pow(
        sum([(cod_value - mean_cod) ** 2 for cod_value in cod_values]) / (
        len(cod_values) - 1), 0.5)
    print('\nmean MSE ± deviation: {:.9f} ± {:.9f}'.format(mean_mse, mse_dev))
    print('mean COD ± deviation: {:.9f} ± {:.9f}'.format(mean_cod, cod_dev))


# Pandas ETL PART
start_time = time.time()
run_etl_pandas()
end_time = time.time()

print('Pandas ETL TIME:', end_time - start_time)

# SDC ETL PART
run_etl_sdc()  # <-- compilation and execution
 
start_time = time.time()
X, y = run_etl_sdc()  # <-- execution only
end_time = time.time()
print('SDC ETL TIME:   ', end_time - start_time)

run_ml(X, y)

