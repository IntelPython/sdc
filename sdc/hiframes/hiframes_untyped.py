# *****************************************************************************
# Copyright (c) 2020, Intel Corporation All rights reserved.
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


from __future__ import print_function, division, absolute_import

import numba

from numba.compiler_machinery import FunctionPass, register_pass

from sdc.hiframes import join, aggregate, sort


@register_pass(mutates_CFG=True, analysis_only=False)
class HiFramesPass(FunctionPass):
    """analyze and transform hiframes calls"""

    _name = "sdc_extention_hi_frames_pass"

    def __init__(self):
        pass

    def run_pass(self, state):
        return HiFramesPassImpl(state).run_pass()


class HiFramesPassImpl(object):

    def __init__(self, state):
        # replace inst variables as determined previously during the pass
        # currently use to keep lhs of Arg nodes intact
        self.replace_var_dict = {}

        # df_var -> {col1:col1_var ...}
        self.df_vars = {}
        # df_var -> label where it is defined
        self.df_labels = {}

        self.arrow_tables = {}
        self.reverse_copies = {}

        # self.state = state
