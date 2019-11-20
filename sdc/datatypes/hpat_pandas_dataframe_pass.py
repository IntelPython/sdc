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

from numba.untyped_passes import InlineClosureLikes
from numba.typed_passes import AnnotateTypes

import sdc


def sdc_nopython_pipeline_lite_register(state, name='nopython'):
    """
    This is to register some sub set of Intel SDC compiler passes in Numba NoPython pipeline
    Each pass, enabled here, is expected to be called many times on every decorated function including
    functions which are not related to Pandas.

    Test: SDC_CONFIG_PIPELINE_SDC=0 python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_sort_values1

    This function needs to be removed if SDC DataFrame support
    no more needs Numba IR transformations via DataFramePass
    """

    if sdc.config.numba_compiler_define_nopython_pipeline_orig is None:
        raise ValueError("Intel SDC. Unexpected usage of DataFrame passes registration function.")

    numba_pass_manager = sdc.config.numba_compiler_define_nopython_pipeline_orig(state, name)

    # numba_pass_manager.add_pass_after(sdc.compiler.InlinePass, InlineClosureLikes)
    # numba_pass_manager.add_pass_after(sdc.hiframes.hiframes_untyped.HiFramesPass, sdc.compiler.InlinePass)
    numba_pass_manager.add_pass_after(sdc.hiframes.hiframes_untyped.HiFramesPass, InlineClosureLikes)

    numba_pass_manager.add_pass_after(sdc.hiframes.dataframe_pass.DataFramePass, AnnotateTypes)
    numba_pass_manager.add_pass_after(sdc.compiler.PostprocessorPass, AnnotateTypes)
    # numba_pass_manager.add_pass_after(sdc.hiframes.hiframes_typed.HiFramesTypedPass, sdc.hiframes.dataframe_pass.DataFramePass)

    numba_pass_manager.finalize()

    return numba_pass_manager
