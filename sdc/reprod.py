import numba
from numba import njit, prange
import numpy
import numpy as np


@njit
def impl(self_data, self_index, other_data, other_index, fill_value=None):
    left_index, right_index = self_index, other_index
    _fill_value = numpy.nan
    joined_index = [0, 1, 2, 3, 3, 4, 9]
    left_indexer = [-1, -1, 4, 0, 2, 1, 3]
    right_indexer = [0, 1, 2, 3, 3, 4, -1]
    result_size = len(joined_index)
    left_values = numpy.empty(result_size, dtype=numpy.float64)
    right_values = numpy.empty(result_size, dtype=numpy.float64)
    for i in numba.prange(result_size):
        left_pos, right_pos = left_indexer[i], right_indexer[i]
        if left_pos != -1:
            left_values[i] = self_data[left_pos]
        else:
            left_values[i] = _fill_value
        if right_pos != -1:
            right_values[i] = other_data[right_pos]
        else:
            right_values[i] = _fill_value
    result_data = left_values + right_values
    return result_data

data = [0, 1, 2, 3, 4]
index = [3, 4, 3, 9, 2]
value = None

print(impl(data, index, index, data, value))
