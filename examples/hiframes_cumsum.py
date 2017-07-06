import pandas as pd
import numpy as np
import hpat

@hpat.jit
def cumsum_df(n):
    df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
    Ac = df.A.cumsum()
    return np.sum(Ac)

n = 10
print(cumsum_df(n))
