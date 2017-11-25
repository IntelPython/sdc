import pandas as pd
import numpy as np
import hpat

@hpat.jit
def shift_df1(n):
    df = pd.DataFrame({'A': np.arange(n)+1.0, 'B': np.random.ranf(n)})
    Ac = df.A.shift(1)
    return Ac.sum()

@hpat.jit
def shift_df2(n):
    df = pd.DataFrame({'A': np.arange(n)+1.0, 'B': np.random.ranf(n)})
    Ac = df.A.pct_change()
    return Ac

n = 10
print("shift 1:")
print(shift_df1(n))
print("pct_change:")
print(shift_df2(n))
