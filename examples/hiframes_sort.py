import pandas as pd
import numpy as np
import hpat

@hpat.jit
def df_sort(df):
    df2 = df.sort_values('A')
    print(df2.A.values)
    print(df2.B.values)

n = 11
df = pd.DataFrame({'A': np.random.ranf(n), 'B': np.arange(n), 'C': np.random.ranf(n)})
# computation is sequential since df is passed in
df_sort(df)
