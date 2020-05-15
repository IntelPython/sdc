import pandas as pd
from numba import njit
import sdc
import vtune as vt
from vtune import task_begin, task_end, domain, string_handle_create
import ctypes
import itt

handle = string_handle_create("Function")


@njit
def dataframe_head(df):
    task_begin(domain, handle)
    series = df['A'].head(n=4)
    task_end(domain)
    return series


df = pd.DataFrame({'A': [1, 2, 4, 6, 4, 2], 'B': [3., 2., 77., 2., 5., 6.5]})

print(dataframe_head(df))
