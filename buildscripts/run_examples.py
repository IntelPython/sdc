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
import traceback

from pathlib import Path
from utilities import SDC_Build_Utilities


EXAMPLES_TO_SKIP = {'basic_usage_nyse_predict.py'}


def run_examples(sdc_utils):
    total = 0
    passed = 0
    failed = 0
    skipped = 0
    failed_examples = []

    os.chdir(str(sdc_utils.examples_path))
    for sdc_example in Path('.').glob('**/*.py'):
        total += 1

        if sdc_example.name in EXAMPLES_TO_SKIP:
            skipped += 1
            continue

        sdc_example = str(sdc_example)
        try:
            sdc_utils.log_info(sdc_utils.line_double)
            sdc_utils.run_command(f'python {str(sdc_example)}')
        except Exception:
            failed += 1
            failed_examples.append(sdc_example)
            sdc_utils.log_info(f'{sdc_example} FAILED')
            traceback.print_exc()
        else:
            passed += 1
            sdc_utils.log_info(f'{sdc_example} PASSED')

    summary_msg = f'SDC examples summary: {total} RUN, {passed} PASSED, {failed} FAILED, {skipped} SKIPPED'
    sdc_utils.log_info(summary_msg, separate=True)
    for failed_example in failed_examples:
        sdc_utils.log_info(f'FAILED: {failed_example}')

    if failed > 0:
        sdc_utils.log_info('Intel SDC examples FAILED', separate=True)
        exit(-1)
    sdc_utils.log_info('Intel SDC examples PASSED', separate=True)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--python', default='3.7', choices=['3.6', '3.7', '3.8'],
                        help='Python version, default = 3.7')
    parser.add_argument('--sdc-channel', default=None, help='Intel SDC channel')

    args = parser.parse_args()

    sdc_utils = SDC_Build_Utilities(args.python, args.sdc_channel)
    sdc_utils.log_info('Run Intel(R) SDC examples', separate=True)
    sdc_utils.log_info(sdc_utils.line_double)
    sdc_utils.create_environment()
    sdc_utils.install_conda_package(['sdc'])

    run_examples(sdc_utils)
