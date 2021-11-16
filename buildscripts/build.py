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


import argparse
import os
import platform

from pathlib import Path
from utilities import SDC_Build_Utilities


def build(sdc_utils):
    os.chdir(str(sdc_utils.src_path))

    sdc_utils.log_info('Start Intel SDC build', separate=True)
    conda_build_cmd = ' '.join([
        'conda build',
        '--no-test',
        f'--python {sdc_utils.python}',
        f'--numpy {sdc_utils.numpy}',
        f'--output-folder {str(sdc_utils.output_folder)}',
        sdc_utils.channels,
        '--override-channels',
        str(sdc_utils.recipe)
    ])
    sdc_utils.run_command(conda_build_cmd)
    sdc_utils.log_info('Intel SDC build SUCCESSFUL', separate=True)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--python', required=True, choices=['3.6', '3.7', '3.8'],
                        help='Python version')
    parser.add_argument('--numpy', required=True, choices=['1.16', '1.17', '1.18'],
                        help='Numpy version')

    args = parser.parse_args()

    sdc_utils = SDC_Build_Utilities(args.python)
    sdc_utils.numpy = args.numpy
    sdc_utils.log_info('Build Intel(R) SDC conda and wheel packages', separate=True)
    sdc_utils.log_info(sdc_utils.line_double)

    sdc_env_packages = ['conda-build']
    if platform.system() == 'Windows':
        sdc_env_packages += ['conda-verify', 'vc', 'vs2015_runtime', 'vs2015_win-64', 'pywin32=223']
    # Install conda-build and other packages from anaconda channel due to issue with wheel
    # output build if use intel channels first
    sdc_utils.create_environment(sdc_env_packages)

    build(sdc_utils)
