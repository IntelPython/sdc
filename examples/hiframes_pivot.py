import pandas as pd
import numpy as np
import hpat


@hpat.jit(pivots={'pt': ['small', 'large']})
def df_pivot(df):
    pt = df.pivot_table(index='A', columns='C', values='D', aggfunc='sum')
    print(pt.small.values)
    print(pt.large.values)
    return


df = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",
                         "bar", "bar", "bar", "bar"],
                   "B": ["one", "one", "one", "two", "two",
                         "one", "one", "two", "two"],
                   "C": ["small", "large", "large", "small",
                         "small", "large", "small", "small",
                         "large"],
                   "D": [1, 2, 2, 6, 3, 4, 5, 6, 9]})

df_pivot(df)
