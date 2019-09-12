import pandas as pd
import numpy as np
import hpat


@hpat.jit
def concat_df(n):
    df1 = pd.DataFrame({'key1': np.arange(n), 'A': np.arange(n) + 1.0})
    df2 = pd.DataFrame({'key2': n - np.arange(n), 'A': n + np.arange(n) + 1.0})
    df3 = pd.concat([df1, df2])
    return df3.key2.sum()


n = 10
print(concat_df(n))
# hpat.distribution_report()
