# *****************************************************************************
# Copyright (c) 2019, Intel Corporation All rights reserved.
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


def jit(signature_or_function=None, **options):

    if 'nopython' not in options:
        '''
        Always use @jit(noPython=True) in SDC by default
        '''
        options['nopython'] = True

    if not sdc.config.config_pipeline_hpat_default:
        '''
        Use Numba compiler pipeline
        '''
        return numba.jit(signature_or_function, **options)

    _locals = options.pop('locals', {})
    assert isinstance(_locals, dict)

    # put pivots in locals TODO: generalize numba.jit options
    pivots = options.pop('pivots', {})
    assert isinstance(pivots, dict)
    for var, vals in pivots.items():
        _locals[var + ":pivot"] = vals

    distributed = set(options.pop('distributed', set()))
    assert isinstance(distributed, (set, list))
    _locals["##distributed"] = distributed

    threaded = set(options.pop('threaded', set()))
    assert isinstance(threaded, (set, list))
    _locals["##threaded"] = threaded

    options['locals'] = _locals

    #options['parallel'] = True
    options['parallel'] = {'comprehension': True,
                           'setitem': False,  # FIXME: support parallel setitem
                           'reduction': True,
                           'numpy': True,
                           'stencil': True,
                           'fusion': True,
                           }

    # Option MPI is boolean and true by default
    # it means MPI transport will be used
    mpi_transport_requested = options.pop('MPI', sdc.config.config_transport_mpi_default)
    if not isinstance(mpi_transport_requested, (int, bool)):
        raise ValueError("Option MPI or SDC_CONFIG_MPI environment variable should be boolean")

    if mpi_transport_requested:
        sdc.config.config_transport_mpi = True
    else:
        sdc.config.config_transport_mpi = False

    return numba.jit(signature_or_function, pipeline_class=sdc.compiler.SDCPipeline, **options)
