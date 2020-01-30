from numba import jit
import numpy as np
import time
import random

import pandas as pd
from sdc.functions.numpy_like import astype

arr_float = np.arange(10 ** 8, dtype=float)
arr_int = np.arange(10 ** 8, dtype=int)


@jit(parallel=True)
def sdc(a, b):
    start_time = time.time()
    res = astype(a, np.int64)
    finish_time = time.time()
    print(res)
    return finish_time - start_time


@jit(parallel=True)
def ref(a, b):
    start_time = time.time()
    res = a.astype(np.int64)
    finish_time = time.time()
    print(res)
    return finish_time - start_time


print(sdc(arr_float, arr_int))
print(ref(arr_float, arr_int))
