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


import argparse
import os
import shutil

from pathlib import Path
from utilities import SDC_Build_Utilities


def run_benchmarks(sdc_utils, args_list, num_threads_list):
    os.chdir(str(sdc_utils.src_path.parent))

    for args_set in args_list:
        for num_threads in num_threads_list:
            os.environ['NUMBA_NUM_THREADS'] = num_threads
            sdc_utils.log_info(f'Run Intel SDC benchmarks on {num_threads} threads', separate=True)
            sdc_utils.run_command(f'python -W ignore -m sdc.runtests {args_set}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--python', default='3.7', choices=['3.6', '3.7', '3.8'],
                        help='Python version, default = 3.7')
    parser.add_argument('--sdc-channel', default=None, help='Intel SDC channel')
    parser.add_argument('--args-list', required=True, nargs='+', help='List of arguments sets for benchmarks')
    parser.add_argument('--num-threads-list', required=True, nargs='+',
                        help='List of values for NUMBA_NUM_THREADS env variable')

    args = parser.parse_args()

    sdc_utils = SDC_Build_Utilities(args.python, args.sdc_channel)
    sdc_utils.log_info('Run Intel(R) SDC benchmarks', separate=True)
    sdc_utils.log_info(sdc_utils.line_double)
    sdc_utils.create_environment(['openpyxl', 'xlrd'])
    sdc_utils.install_conda_package(['sdc'])

    run_benchmarks(sdc_utils, args.args_list, args.num_threads_list)
