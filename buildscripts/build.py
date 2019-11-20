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
from utilities import set_environment_variable


def run_smoke_tests(sdc_src, test_env_activate):
        sdc_pi_example = os.path.join(sdc_src, 'buildscripts', 'sdc_pi_example.py')
        run_command(f'{test_env_activate} && python -c "import sdc"')
        run_command(f'{test_env_activate} && python {sdc_pi_example}')


if __name__ == '__main__':
    sdc_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sdc_recipe = os.path.join(sdc_src, 'buildscripts', 'sdc-conda-recipe')
    numba_recipe = os.path.join(sdc_src, 'buildscripts', 'numba-conda-recipe', 'recipe')
    numba_output_folder = os.path.join(sdc_src, 'numba-build')
    numba_master_channel = f'file:/{numba_output_folder}'
    if platform.system() == 'Windows':
        numba_master_channel = f'{numba_output_folder}'

    os.chdir(sdc_src)

    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-mode', default='develop', choices=['develop', 'install', 'package'],
                        help="""Build mode:
                                develop: install package in source directory (default)
                                install: build and install package in build environment
                                package: build conda and wheel packages""")
    parser.add_argument('--python', default='3.7', choices=['3.6', '3.7', '3.8'],
                        help='Python version to build with, default = 3.7')
    parser.add_argument('--numpy', default='1.16', choices=['1.15', '1.16', '1.17'],
                        help='Numpy version to build with, default = 1.16')
    parser.add_argument('--output-folder', default=os.path.join(sdc_src, 'sdc-build'),
                        help='Output folder for build packages, default = sdc-build')
    parser.add_argument('--conda-prefix', default=None, help='Conda prefix')
    parser.add_argument('--skip-smoke-tests', action='store_true', help='Skip smoke tests for build')
    parser.add_argument('--numba-channel', default='numba',
                        help='Numba channel to build with special Numba, default=numba')
    parser.add_argument('--use-numba-master', action='store_true', help=f'Build with Numba from master')
    parser.add_argument('--channel-list', default=None, help='List of channels to use: "-c <channel> -c <channel>"')

    args = parser.parse_args()

    build_mode = args.build_mode
    python = args.python
    numpy = args.numpy
    output_folder = args.output_folder
    conda_prefix = os.getenv('CONDA_PREFIX', args.conda_prefix)
    skip_smoke_tests = args.skip_smoke_tests
    channel_list = args.channel_list
    use_numba_master = args.use_numba_master
    numba_channel = numba_master_channel if use_numba_master is True else args.numba_channel
    assert conda_prefix is not None, 'CONDA_PREFIX is not defined; Please use --conda-prefix option or activate your conda'
    
    # Init variables
    conda_activate = get_conda_activate_cmd(conda_prefix).replace('"', '')
    build_env = f'sdc-build-env-py{python}-numpy{numpy}'
    test_env = f'sdc-test-env-py{python}-numpy{numpy}'
    develop_env = f'sdc-develop-env-py{python}-numpy{numpy}'
    build_env_activate = get_activate_env_cmd(conda_activate, build_env)
    test_env_activate = get_activate_env_cmd(conda_activate, test_env)
    develop_env_activate = get_activate_env_cmd(conda_activate, develop_env)

    conda_channels = f'-c {numba_channel} -c conda-forge -c intel -c defaults --override-channels'
    numba_conda_channels = '-c conda-forge -c intel --override-channels'
    if channel_list:
        conda_channels = f'{channel_list} --override-channels'

    conda_build_packages = ['conda-build']
    if platform.system() == 'Windows':
        if build_mode != 'package':
            set_environment_variable('INCLUDE', os.path.join('%CONDA_PREFIX%', 'Library', 'include'))
            set_environment_variable('LIB', os.path.join('%CONDA_PREFIX%', 'Library', 'lib'))

        conda_build_packages.extend(['conda-verify', 'vc', 'vs2015_runtime', 'vs2015_win-64'])

    # Build Numba from master
    if use_numba_master is True:
        create_conda_env(conda_activate, build_env, python, conda_build_packages)
        format_print('Start build Numba from master')
        run_command('{} && {}'.format(build_env_activate,
                                      ' '.join(['conda build --no-test',
                                                f'--python {python}',
                                                f'--numpy {numpy}',
                                                f'--output-folder {numba_output_folder}',
                                                f'{numba_conda_channels} {numba_recipe}'])))
        format_print('NUMBA BUILD COMPETED')

    # Get sdc build and test environment
    sdc_env = get_sdc_env(conda_activate, sdc_src, sdc_recipe, python, numpy, conda_channels)

    # Set build command
    if build_mode == 'package':
        create_conda_env(conda_activate, build_env, python, conda_build_packages)
        build_cmd = '{} && {}'.format(build_env_activate,
                                      ' '.join(['conda build --no-test',
                                                            f'--python {python}',
                                                            f'--numpy {numpy}',
                                                            f'--output-folder {output_folder}',
                                                            f'{conda_channels} {sdc_recipe}']))
    else:
        create_conda_env(conda_activate, develop_env, python, sdc_env['build'], conda_channels)
        build_cmd = f'{develop_env_activate} && python setup.py {build_mode}'

    # Start build
    format_print('START SDC BUILD')
    run_command(build_cmd)
    format_print('BUILD COMPETED')

    # Check if smoke tests should be skipped
    if skip_smoke_tests is True:
        format_print('Smoke tests are skipped due to "--skip-smoke-tests" option')
        sys.exit(0)

    # Start smoke tests
    format_print('Run Smoke tests')
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

    if build_mode == 'package':
        # Get build packages
        sdc_packages = get_sdc_build_packages(output_folder)
        for package in sdc_packages:
            if '.tar.bz2' in package:
                # Start test for conda package
                format_print(f'Run tests for sdc conda package: {package}')
                create_conda_env(conda_activate, test_env, python, sdc_env['test'], conda_channels)
                run_command(f'{test_env_activate} && conda install -y {package}')
                run_smoke_tests(sdc_src, test_env_activate)
            elif '.whl' in package:
                # Start test for wheel package
                format_print(f'Run tests for sdc wheel package: {package}')
                create_conda_env(conda_activate, test_env, python, sdc_env['test'] + ['pip'], conda_channels)
                run_command(f'{test_env_activate} && pip install {package}')
                run_smoke_tests(sdc_src, test_env_activate)
    else:
        format_print('Run tests for installed sdc package')
        run_smoke_tests(sdc_src, develop_env_activate)

    format_print('SMOKE TESTS ARE PASSED')
