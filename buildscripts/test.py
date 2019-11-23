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



import argparse
import os
import platform
import subprocess
import sys
import traceback

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
    parser.add_argument('--test-mode', default='conda', choices=['conda', 'package', 'develop'],
                        help="""Test mode:
                        conda:   use conda-build to run tests (default and valid for conda package-type)
                        package: create test environment, install package there and run tests
                        develop: run tests for sdc package already installed in develop mode""")
    parser.add_argument('--package-type', default='conda', choices=['conda', 'wheel'],
                        help='Package to test: conda or wheel, default = conda')
    parser.add_argument('--python', default='3.7', choices=['3.6', '3.7', '3.8'],
                        help='Python version to test with, default = 3.7')
    parser.add_argument('--numpy', default='1.16', choices=['1.15', '1.16', '1.17'],
                        help='Numpy version to test with, default = 1.16')
    parser.add_argument('--build-folder', default=os.path.join(sdc_src, 'sdc-build'),
                        help='Built packages location, default = sdc-build')
    parser.add_argument('--conda-prefix', default=None, help='Conda prefix')
    parser.add_argument('--run-coverage', default='False', choices=['True', 'False'],
                        help='Run coverage (sdc must be build in develop mode)')
    parser.add_argument('--numba-channel', default='numba',
                        help='Numba channel to build with special Numba, default=numba')
    parser.add_argument('--use-numba-master', action='store_true',
                        help=f'Test with Numba master from {numba_master_channel}')
    parser.add_argument('--channel-list', default=None, help='List of channels to use: "-c <channel> -c <channel>"')

    args = parser.parse_args()

    test_mode = args.test_mode
    package_type = args.package_type
    python = args.python
    numpy = args.numpy
    build_folder = args.build_folder
    conda_prefix = os.getenv('CONDA_PREFIX', args.conda_prefix)
    run_coverage = args.run_coverage
    channel_list = args.channel_list
    use_numba_master = args.use_numba_master
    numba_channel = numba_master_channel if use_numba_master is True else args.numba_channel
    assert conda_prefix is not None, 'CONDA_PREFIX is not defined; Please use --conda-prefix option or activate your conda'

    # Init variables
    conda_activate = get_conda_activate_cmd(conda_prefix).replace('"', '')
    test_env = f'sdc-test-env-py{python}-numpy{numpy}'
    develop_env = f'sdc-develop-env-py{python}-numpy{numpy}'
    test_env_activate = get_activate_env_cmd(conda_activate, test_env)
    develop_env_activate = get_activate_env_cmd(conda_activate, develop_env)

    conda_channels = f'-c {numba_channel} -c conda-forge -c intel -c defaults --override-channels'
    if channel_list:
        conda_channels = f'{channel_list} --override-channels'

    if platform.system() == 'Windows':
        test_script = os.path.join(sdc_recipe, 'run_test.bat')
    else:
        test_script = os.path.join(sdc_recipe, 'run_test.sh')


    if run_coverage == 'True':
        if platform.system() == 'Windows':
            format_print('Coverage can be run only on Linux of mac for now')
            sys.exit(0)

        os.environ['HDF5_DIR'] = conda_prefix
        coverage_omit = './sdc/ml/*,./sdc/cv_ext.py,./sdc/tests/*'
        coverage_cmd = ' && '.join(['coverage erase',
                                   f'coverage run --source=./sdc --omit {coverage_omit} ./sdc/runtests.py',
                                   'coveralls -v'])

        format_print('Run coverage')
        format_print(f'Assume that SDC is installed in develop build-mode to {develop_env} environment', new_block=False)
        format_print('Install scipy and coveralls')
        run_command(f'{develop_env_activate} && conda install -q -y scipy coveralls')
        run_command(f'{develop_env_activate} && python -m sdc.tests.gen_test_data')
        run_command(f'{develop_env_activate} && {coverage_cmd}')
        sys.exit(0)

    if test_mode == 'develop':
        format_print(f'Run tests for sdc installed to {develop_env}')
        """
        os.chdir(../sdc_src) is a workaround for the following error:
        Traceback (most recent call last):
            File "<string>", line 1, in <module>
            File "sdc/sdc/__init__.py", line 9, in <module>
                import sdc.dict_ext
            File "sdc/sdc/dict_ext.py", line 12, in <module>
                from sdc.str_ext import string_type, gen_unicode_to_std_str, gen_std_str_to_unicode
            File "sdc/sdc/str_ext.py", line 18, in <module>
                from . import hstr_ext
        ImportError: cannot import name 'hstr_ext' from 'sdc' (sdc/sdc/__init__.py)
        """
        os.chdir(os.path.dirname(sdc_src))
        run_command(f'{develop_env_activate} && {test_script}')
        format_print('Tests for installed SDC package are PASSED')
        sys.exit(0)

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

    # Test specified package type
    if test_mode == 'package':
        format_print(f'Run tests for {package_type} package type')
        """
        os.chdir(../sdc_src) is a workaround for the following error:
        Traceback (most recent call last):
            File "<string>", line 1, in <module>
            File "sdc/sdc/__init__.py", line 9, in <module>
                import sdc.dict_ext
            File "sdc/sdc/dict_ext.py", line 12, in <module>
                from sdc.str_ext import string_type, gen_unicode_to_std_str, gen_std_str_to_unicode
            File "sdc/sdc/str_ext.py", line 18, in <module>
                from . import hstr_ext
        ImportError: cannot import name 'hstr_ext' from 'sdc' (sdc/sdc/__init__.py)
        """
        os.chdir(os.path.dirname(sdc_src))
        sdc_packages = get_sdc_build_packages(build_folder)
        for package in sdc_packages:
            if '.tar.bz2' in package and package_type == 'conda':
                format_print(f'Run tests for sdc conda package: {package}')
                create_conda_env(conda_activate, test_env, python, sdc_env['test'], conda_channels)
                run_command(f'{test_env_activate} && conda install -y {package}')
                run_command(f'{test_env_activate} && {test_script}')
            elif '.whl' in package and package_type == 'wheel':
                format_print(f'Run tests for sdc wheel package: {package}')
                create_conda_env(conda_activate, test_env, python, sdc_env['test'] + ['pip'], conda_channels)
                run_command(f'{test_env_activate} && pip install {package}')
                run_command(f'{test_env_activate} && {test_script}')

        format_print(f'Tests for {package_type} packages are PASSED')
