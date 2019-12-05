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
This is a set of configuration variables in SDC initialized at startup
'''


import os
from distutils import util as distutils_util

try:
    import pyarrow
except ImportError:
    _has_pyarrow = False
else:
    _has_pyarrow = True

try:
    from . import cv_wrapper
except ImportError:
    _has_opencv = False
else:
    _has_opencv = True
    import sdc.cv_ext

config_transport_mpi_default = distutils_util.strtobool(os.getenv('SDC_CONFIG_MPI', 'True'))
'''
Default value for transport used if no function decorator controls the transport
'''

config_transport_mpi = config_transport_mpi_default
'''
Current value for transport controlled by decorator need to initialize this here
because decorator called later then modules have been initialized
'''

config_pipeline_hpat_default = distutils_util.strtobool(os.getenv('SDC_CONFIG_PIPELINE_SDC', 'True'))
'''
Default value used to select compiler pipeline in a function decorator
'''

if not config_pipeline_hpat_default:
    # avoid using MPI transport if no SDC compiler pipeline used
    config_transport_mpi_default = False
    config_transport_mpi = config_transport_mpi_default

numba_compiler_define_nopython_pipeline_orig = None
'''
Default value for a pointer intended to use as Numba.DefaultPassBuilder.define_nopython_pipeline() in overloaded function
'''
