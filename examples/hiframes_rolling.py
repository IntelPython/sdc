import pandas as pd
import numpy as np
import hpat


@hpat.jit
def rolling_df1(n):
    df = pd.DataFrame({'A': np.arange(n), 'B': np.random.ranf(n)})
    Ac = df.A.rolling(5).sum()
    return Ac.sum()


@hpat.jit
def rolling_df2(n):
    df = pd.DataFrame({'A': np.arange(n), 'B': np.random.ranf(n)})
    df['moving average'] = df.A.rolling(window=5, center=True).mean()
    return df['moving average'].sum()


@hpat.jit
def rolling_df3(n):
    df = pd.DataFrame({'A': np.arange(n), 'B': np.random.ranf(n)})
    Ac = df.A.rolling(3, center=True).apply(lambda a: a[0] + 2 * a[1] + a[2])
    return Ac.sum()


n = 10
print("sum left window:")
print(rolling_df1(n))
print("mean center window:")
print(rolling_df2(n))
print("custom center window:")
print(rolling_df3(n))
