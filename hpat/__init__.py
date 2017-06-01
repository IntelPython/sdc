import numba
from numba import *

def jit(signature_or_function=None, **options):
    # set nopython by default
    if 'nopython' not in options:
        options['nopython'] = True
    return numba.jit(signature_or_function, **options)
