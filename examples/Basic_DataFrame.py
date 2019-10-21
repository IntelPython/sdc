import pandas as pd
import numpy as np
import hpat


@hpat.jit
def merge_df(n):
    df1 = pd.DataFrame({'key1': np.arange(n), 'A': np.arange(n) + 1.0})
    df2 = pd.DataFrame({'key2': n - np.arange(n), 'B': n + np.arange(n) + 1.0})
    df3 = pd.merge(df1, df2, left_on='key1', right_on='key2')
    return df3


@hpat.jit
def concat_df(n):
    df1 = pd.DataFrame({'key1': np.arange(n), 'A': np.arange(n) + 1.0})
    df2 = pd.DataFrame({'key2': n - np.arange(n), 'A': n + np.arange(n) + 2.0})
    df3 = pd.concat([df1, df2])
    return df3


n = 10
print("the merged output is:")
print(merge_df(n))
print("==================================")
print("concat output is:")
print(concat_df(n))
