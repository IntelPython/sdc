# *****************************************************************************
# Copyright (c) 2019-2021, Intel Corporation All rights reserved.
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
import shutil
import traceback
import re

from pathlib import Path
from utilities import SDC_Build_Utilities


def check_sdc_installed(sdc_utils, sdc_package):
    cmd_output = sdc_utils.get_command_output('conda list sdc')
    pattern = sdc_package.replace('=', r'\s+')
    return re.search(pattern, cmd_output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--python', default='3.7', choices=['3.6', '3.7', '3.8'],
                        help='Python version, default = 3.7')
    parser.add_argument('--channels', default=None, help='Default env channels')
    parser.add_argument('--sdc-channel', default=None, help='Intel SDC channel')

    args = parser.parse_args()

    sdc_utils = SDC_Build_Utilities(args.python, args.channels, args.sdc_channel)
    sdc_utils.log_info('Test Intel(R) SDC conda install', separate=True)
    sdc_utils.log_info(sdc_utils.line_double)
    sdc_utils.create_environment()
    sdc_package = f'sdc={sdc_utils.get_sdc_version_from_channel()}'

    # channels list is aligned with install instruction in README.rst
    install_channels = "-c intel/label/beta -c intel -c defaults -c conda-forge"
    sdc_utils.install_conda_package([sdc_package], channels=install_channels)

    assert check_sdc_installed(sdc_utils, sdc_package), "SDC package was not installed"
