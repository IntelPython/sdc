import pandas as pd
import numpy as np
import hpat

@hpat.jit
def shift_df(n):
    df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
    Ac = df.A.shift(1)
    return Ac.sum()

n = 10
print(shift_df(n))
