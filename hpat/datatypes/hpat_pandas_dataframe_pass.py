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
| Procedures are required for SDC DataFrameType handling in Numba
'''

import hpat


def sdc_dataframepassimpl_overload(*args, **kwargs):
    """
    This is a pointer intended to use as Numba AnnotateTypes run_pass() function
    A hook made to overload Numba function and:
    - call original function
    - call DataFramePass exposed in this module

    return the status of original Numba function

    This function needs to be removed if SDC DataFrame support
    no more needs Numba IR transformations via DataFramePass
    """

    if hpat.config.numba_typed_passes_annotatetypes_orig is None:
        """
        Unexpected usage of this function
        """

        return False

    status_numba_pass = hpat.config.numba_typed_passes_annotatetypes_orig(*args, **kwargs)

    numba_state_var = args[1]

    status_dataframe_pass = hpat.hiframes.dataframe_pass.DataFramePassImpl(numba_state_var).run_pass()
    status_postprocess_pass = hpat.compiler.PostprocessorPass().run_pass(numba_state_var)
    status_dataframe_typed_pass = hpat.hiframes.hiframes_typed.HiFramesTypedPassImpl(numba_state_var).run_pass()

    is_ir_mutated = status_numba_pass or status_dataframe_pass or status_postprocess_pass or status_dataframe_typed_pass

    return is_ir_mutated
