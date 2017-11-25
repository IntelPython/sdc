import pandas as pd
import numpy as np
import hpat

@hpat.jit
def cumsum_df(n):
    df = pd.DataFrame({'A': np.arange(n)+1.0, 'B': np.random.ranf(n)})
    Ac = df.A.cumsum()
    return Ac.sum()

n = 10
print(cumsum_df(n))
