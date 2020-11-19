# *****************************************************************************
# Copyright (c) 2019-2020, Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

'''
This is a function decorator definition
'''

import numba
import sdc

from functools import wraps
from sdc.utilities.utils import print_compile_times


def jit(signature_or_function=None, **options):

    if 'nopython' not in options:
        '''
        Always use @jit(noPython=True) in SDC by default
        '''
        options['nopython'] = True

    '''
    Use Numba compiler pipeline
    '''
    return numba.jit(signature_or_function, **options)


def debug_compile_time(level=1, func_names=None):
    """ Decorates Numba Dispatcher object to print compile stats after call.
        Usage:
            @debug_compile_time()
            @numba.njit
            <decorated function>
        Args:
            level: if zero prints only short summary
            func_names: filters output to include only functions which names include listed strings,
    """

    def get_wrapper(disp):

        @wraps(disp)
        def wrapper(*args, **kwargs):
            res = disp(*args, **kwargs)
            print('*' * 40, 'COMPILE STATS', '*' * 40)
            print_compile_times(disp, level=level, func_names=func_names)
            print('*' * 95)
            return res

        return wrapper

    return get_wrapper
