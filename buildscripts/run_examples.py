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
import multiprocessing as mp


EXAMPLES_TO_SKIP = {'basic_usage_nyse_predict.py'}
TEST_TIMEOUT = 120


# keep test results global to be visible for async callbacks
class TestResults():
    total = 0
    failed = 0
    passed = 0
    skipped = 0
    failed_examples = []


def run_single_example(path, sdc_utils):
    str_path = str(path)
    try:
        sdc_utils.log_info(sdc_utils.line_double)
        sdc_utils.run_command(f'python {str_path}')
    except Exception as e:
        raise Exception(str_path).with_traceback(e.__traceback__)

    return str_path


def normal_handler(test_name):
    TestResults.passed += 1
    sdc_utils.log_info(f'{test_name} PASSED')


def error_handler(error):
    TestResults.failed += 1
    test_name = str(error).split()[-1]
    sdc_utils.log_info(f'{test_name} FAILED')
    TestResults.failed_examples.append(test_name)


def run_examples(sdc_utils):

    os.chdir(str(sdc_utils.examples_path))
    pool = mp.Pool(max(1, mp.cpu_count()))

    task_queue = []
    for sdc_example in Path('.').glob('**/*.py'):
        TestResults.total += 1

        if sdc_example.name in EXAMPLES_TO_SKIP:
            TestResults.skipped += 1
            continue

        sdc_example = str(sdc_example)
        task_queue.append(pool.apply_async(
            run_single_example,
            [sdc_example, sdc_utils],
            callback=normal_handler,
            error_callback=error_handler
        ))

    for promise in task_queue:
        try:
            promise.get(TEST_TIMEOUT)
        except Exception:
            traceback.print_exc()

    pool.close()
    pool.join()

    summary_msg = f'SDC examples summary: {TestResults.total} RUN, {TestResults.passed} PASSED, ' \
                  f'{TestResults.failed} FAILED, {TestResults.skipped} SKIPPED'
    sdc_utils.log_info(summary_msg, separate=True)
    for test_name in TestResults.failed_examples:
        sdc_utils.log_info(f'FAILED: {test_name}')

    if TestResults.failed > 0:
        sdc_utils.log_info('Intel SDC examples FAILED', separate=True)
        exit(-1)
    sdc_utils.log_info('Intel SDC examples PASSED', separate=True)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--python', default='3.7', choices=['3.6', '3.7', '3.8'],
                        help='Python version, default = 3.7')
    parser.add_argument('--channels', default=None, help='Default env channels')
    parser.add_argument('--sdc-channel', default=None, help='Intel SDC channel')

    args = parser.parse_args()

    sdc_utils = SDC_Build_Utilities(args.python, args.channels, args.sdc_channel)
    sdc_utils.log_info('Run Intel(R) SDC examples', separate=True)
    sdc_utils.log_info(sdc_utils.line_double)
    sdc_utils.create_environment()
    sdc_package = f'sdc={sdc_utils.get_sdc_version_from_channel()}'
    sdc_utils.install_conda_package([sdc_package])

    run_examples(sdc_utils)
