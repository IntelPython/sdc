import pandas as pd
import numpy as np

### 



### Let's define use-cases first, what we actually need from multi-index
### is having it as DF columns! that is we need to support indexes
### that arise from groupby.agg method.

def test_impl_1(df):
    A = df.groupby('A').agg({'A': ['count', 'min', 'max'],
                             'B': ['std', 'mean']})
    return A

df = pd.DataFrame({
    'A': [2, 1, 1, 1, 2, 2, 1],
    'B': [-8, 2, 3, 1, 5, 6, 7]
})

# print("df:", df)
# res = test_impl_1(df)
# print("res:", res)



def test_impl_2(df):
    A = df.groupby('A').agg([lambda x: x.max() - x.min(), lambda x: x.max() + x.min()])
    return A

df = pd.DataFrame({
    'A': [2, 1, 1, 1, 2, 2, 1],
    'B': [-8, 2, 3, 1, 5, 6, 7],
    'C': [-81, 21, 31, 11, 51, 61, 71]
})

# print("df:", df)
# res = test_impl_2(df)
# print("res:", res)


def test_impl_3():
    res = pd.MultiIndex(
        levels=[np.array([1, 2]), np.array([3, 4])],
        #levels=[["zero", "one"], ["x", "y"]],
        codes=[[1, 1, 0, 0], [1, 0, 1, 0]]
    )
    return res

res = test_impl_3()
print("res:", res)
