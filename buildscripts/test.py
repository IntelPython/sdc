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
import platform
import re
import subprocess
import sys
import traceback

from pathlib import Path

from utilities import create_conda_env
from utilities import format_print
from utilities import get_sdc_env
from utilities import get_sdc_build_packages
from utilities import get_activate_env_cmd
from utilities import get_conda_activate_cmd
from utilities import run_command


if __name__ == '__main__':
    sdc_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sdc_recipe = os.path.join(sdc_src, 'buildscripts', 'sdc-conda-recipe')
    numba_output_folder = os.path.join(sdc_src, 'numba-build')
    numba_master_channel = f'file:/{numba_output_folder}'
    if platform.system() == 'Windows':
        numba_master_channel = f'{numba_output_folder}'

    os.chdir(sdc_src)

    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-mode', default='conda', choices=['conda', 'benchmark', 'examples'],
                        help="""Test mode:
                        conda:     use conda-build to run tests (default)
                        benchmark: run benchmark tests for sdc
                        examples:  execute examples, located in sdc/examples""")
    parser.add_argument('--python', default='3.7', choices=['3.6', '3.7', '3.8'],
                        help='Python version to test with, default = 3.7')
    parser.add_argument('--numpy', default='1.17', choices=['1.16', '1.17'],
                        help='Numpy version to test with, default = 1.17')
    parser.add_argument('--build-folder', default=os.path.join(sdc_src, 'sdc-build'),
                        help='Built packages location, default = sdc-build')
    parser.add_argument('--conda-prefix', default=None, help='Conda prefix')
    parser.add_argument('--numba-channel', default='numba',
                        help='Numba channel to build with special Numba, default=numba')
    parser.add_argument('--use-numba-master', action='store_true',
                        help=f'Test with Numba master from {numba_master_channel}')
    parser.add_argument('--channel-list', default=None, help='List of channels to use: "-c <channel> -c <channel>"')
    parser.add_argument('--benchmark-test-module', default='sdc.tests.tests_perf',
                        help='Run performance testing for all or a sigle one"')
    parser.add_argument('--benchmark-num-threads-list', nargs='+',
                        help='List of values for NUMBA_NUM_THREADS env variable', type=int)

    args = parser.parse_args()

    test_mode = args.test_mode
    python = args.python
    numpy = args.numpy
    build_folder = args.build_folder
    conda_prefix = os.getenv('CONDA_PREFIX', args.conda_prefix)
    channel_list = args.channel_list
    use_numba_master = args.use_numba_master
    benchmark_test_module = args.benchmark_test_module
    benchmark_num_threads_list = args.benchmark_num_threads_list
    numba_channel = numba_master_channel if use_numba_master is True else args.numba_channel
    assert conda_prefix is not None, 'CONDA_PREFIX is not defined; Please use --conda-prefix option or activate your conda'

    # Init variables
    conda_activate = get_conda_activate_cmd(conda_prefix).replace('"', '')
    test_env = f'sdc-test-env-py{python}-numpy{numpy}'
    test_env_activate = get_activate_env_cmd(conda_activate, test_env)

    conda_channels = f'-c {numba_channel} -c defaults -c conda-forge --override-channels'
    # If numba is taken from custom channel, need to add numba channel to get dependencies
    if numba_channel != 'numba':
        conda_channels = f'-c {numba_channel} -c numba -c defaults -c conda-forge --override-channels'
    if channel_list:
        conda_channels = f'{channel_list} --override-channels'

    if platform.system() == 'Windows':
        test_script = os.path.join(sdc_recipe, 'run_test.bat')
    else:
        test_script = os.path.join(sdc_recipe, 'run_test.sh')

    # Test conda package using conda build
    if test_mode == 'conda':
        create_conda_env(conda_activate, test_env, python, ['conda-build'])
        sdc_packages = get_sdc_build_packages(build_folder)
        for package in sdc_packages:
            if '.tar.bz2' in package:
                format_print(f'Run tests for sdc conda package: {package}')
                run_command(f'{test_env_activate} && conda build --test {conda_channels} {package}')
        format_print('Tests for conda packages are PASSED')
        sys.exit(0)

    # Get sdc build and test environment
    sdc_env = get_sdc_env(conda_activate, sdc_src, sdc_recipe, python, numpy, conda_channels)

    # Execute examples
    if test_mode == 'examples':
        format_print('Run SDC examples')
        sdc_examples = os.path.join(sdc_src, 'examples')
        passed_examples = []
        failed_examples = []
        expected_failures = []
        unexpected_success = []
        expected_failures_list = []

        os.chdir(sdc_examples)
        sdc_packages = get_sdc_build_packages(build_folder)
        for package in sdc_packages:
            if '.tar.bz2' in package:
                format_print(f'Run examples for sdc conda package: {package}')
                create_conda_env(conda_activate, test_env, python, sdc_env['test'], conda_channels)
                run_command(f'{test_env_activate} && conda install -y {package}')
                for item in Path('.').glob('**/*.py'):
                    item = str(item)
                    if 'old_examples' in item:
                        continue

                    format_print(f'Execute {item}')
                    try:
                        run_command(f'{test_env_activate} && python {item}')
                        if item in expected_failures_list:
                            unexpected_success.append(item)
                    except Exception:
                        if item in expected_failures_list:
                            expected_failures.append(item)
                        else:
                            failed_examples.append(item)
                        format_print(f'{item} FAILED', new_block=False)
                        traceback.print_exc()
                    else:
                        format_print(f'{item} PASSED', new_block=False)
                        passed_examples.append(item)

        total_passed = len(passed_examples)
        total_failed = len(failed_examples)
        total_expected_failures = len(expected_failures)
        total_unexpected_success = len(unexpected_success)
        total_run = total_passed + total_failed + total_expected_failures + total_unexpected_success
        format_print(' '.join([f'SDC examples summary:',
                               f'{total_run} RUN,',
                               f'{total_passed} PASSED,',
                               f'{total_failed} FAILED,',
                               f'{total_expected_failures} EXPECTED FAILURES',
                               f'{total_unexpected_success} UNEXPECTED SUCCESS',
                               ]))
        for item in passed_examples:
            format_print(f' - {item}: PASSED', new_block=False)
        for item in failed_examples:
            format_print(f' - {item}: FAILED', new_block=False)
        for item in expected_failures:
            format_print(f' - {item}: EXPECTED FAILED', new_block=False)
        for item in unexpected_success:
            format_print(f' - {item}: UNEXPECTED SUCCESS', new_block=False)

        success = (total_failed + total_unexpected_success) == 0
        sys.exit(0 if success else -1)

    # Benchmark tests
    if test_mode == 'benchmark':
        if benchmark_num_threads_list is None:
            msg = ("List of values for NUMBA_NUM_THREADS is not defined; "
                   "Please use --benchmark-num-threads-list option")
            raise ValueError(msg)
        os.chdir(os.path.dirname(sdc_src))
        sdc_packages = get_sdc_build_packages(build_folder)
        for package in sdc_packages:
            if '.tar.bz2' in package:
                format_print(f'Run benchmark tests for sdc conda package: {package}')
                create_conda_env(conda_activate, test_env, python, sdc_env['test'] + ['openpyxl', 'xlrd', 'pytest-xdist', 'pytest-timeout'],
                                 conda_channels)
                run_command(f'{test_env_activate} && conda install -y {package}')
                for num_threads in benchmark_num_threads_list:
                    os.environ['NUMBA_NUM_THREADS'] = str(num_threads)
                    format_print(f'NUMBA_NUM_THREADS is : {num_threads}')
                    run_command(f'{test_env_activate} && python -W ignore -m pytest --timeout=600 --timeout-method=thread --pyargs {benchmark_test_module}')
