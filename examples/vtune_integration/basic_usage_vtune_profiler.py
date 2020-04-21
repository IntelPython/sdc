import pandas as pd
from numba import njit
import sdc
import sdc.vtune_integration as vt
import ctypes
import itt

handle = vt.ctypes_string_handle_create(b"Head\0")

functype_task_begin = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)
ctypes_task_begin = functype_task_begin(itt.__itt_task_begin)

functype_task_end = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
ctypes_task_end = functype_task_end(itt.__itt_task_end)


@njit
def dataframe_head(df):
    ctypes_task_begin(vt.domain, handle)
    new_df = df.head(n=5)
    ctypes_task_end(vt.domain)
    return new_df


df = pd.DataFrame({'A': [1,2,4,6,4,2], 'B': [3.,2.,77.,2.,5.,6.5]})

print(dataframe_head(df))
