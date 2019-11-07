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
from utilities import setup_conda


def run_smoke_tests(sdc_src, test_env_activate):
        sdc_pi_example = os.path.join(sdc_src, 'buildscripts', 'sdc_pi_example.py')
        run_command(f'{test_env_activate} && python -c "import hpat"')
        run_command(f'{test_env_activate} && python {sdc_pi_example}')


if __name__ == '__main__':
    sdc_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sdc_recipe = os.path.join(sdc_src, 'buildscripts', 'sdc-conda-recipe')

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

    args = parser.parse_args()

    build_mode       = args.build_mode
    python           = args.python
    numpy            = args.numpy
    output_folder    = args.output_folder
    conda_prefix     = os.getenv('CONDA_PREFIX', args.conda_prefix)
    skip_smoke_tests = args.skip_smoke_tests
    assert conda_prefix is not None, 'CONDA_PREFIX is not defined; Please use --conda-prefix option or activate your conda'
    
    # Init variables
    conda_activate       = get_conda_activate_cmd(conda_prefix).replace('"', '')
    build_env            = f'sdc-build-env-py{python}-numpy{numpy}'
    test_env             = f'sdc-test-env-py{python}-numpy{numpy}'
    develop_env          = f'sdc-develop-env-py{python}-numpy{numpy}'
    build_env_activate   = get_activate_env_cmd(conda_activate, build_env)
    test_env_activate    = get_activate_env_cmd(conda_activate, test_env)
    develop_env_activate = get_activate_env_cmd(conda_activate, develop_env)

    conda_channels = '-c numba -c conda-forge -c defaults -c intel'

    conda_build_packages = ['conda-build']
    if platform.system() == 'Windows':
        set_environment_variable('INCLUDE', os.path.join('%CONDA_PREFIX%', 'Library', 'include'))
        set_environment_variable('LIB', os.path.join('%CONDA_PREFIX%', 'Library', 'lib'))
        """
        For develop build vs-2015 and vs-2017 runtime is installed.
        If Visual Studio 2017 is not installed, activation returns non-zero code
        thus next command executed with && fails.
        This delimited is used for develop build and after-build smoke tests.
        """
        conda_build_packages.extend(['conda-verify', 'vc', 'vs2015_runtime', 'vs2015_win-64'])

    # Setup conda
    setup_conda(conda_activate)

    # Get sdc build and test environment
    sdc_env = get_sdc_env(conda_activate, sdc_src, sdc_recipe, python, numpy, conda_channels)

    # Set build command
    if build_mode == 'package':
        create_conda_env(conda_activate, build_env, python, conda_build_packages)
        build_cmd = '{} && {}'.format(build_env_activate,
                                      ' '.join(['conda build --no-test',
                                                            f'--python {python}',
                                                            f'--numpy={numpy}',
                                                            f'--output-folder {output_folder}',
                                                            f'--prefix-length 10 {sdc_recipe}']))
    else:
        create_conda_env(conda_activate, develop_env, python, sdc_env['build'])
        build_cmd = f'{develop_env_activate} && python setup.py {build_mode}'

    # Start build
    format_print('Start build')
    run_command(build_cmd)
    format_print('BUILD COMPETED')

    # Check if smoke tests should be skipped
    if skip_smoke_tests == True:
        format_print('Smoke tests are skipped due to "--skip-smoke-tests" option')
        sys.exit(0)

    # Start smoke tests
    format_print('Run Smoke tests')
    """
    os.chdir(../sdc_src) is a workaround for the following error:
    Traceback (most recent call last):
        File "<string>", line 1, in <module>
        File "hpat/hpat/__init__.py", line 9, in <module>
            import hpat.dict_ext
        File "hpat/hpat/dict_ext.py", line 12, in <module>
            from hpat.str_ext import string_type, gen_unicode_to_std_str, gen_std_str_to_unicode
        File "hpat/hpat/str_ext.py", line 18, in <module>
            from . import hstr_ext
    ImportError: cannot import name 'hstr_ext' from 'hpat' (hpat/hpat/__init__.py)
    """
    os.chdir(os.path.dirname(sdc_src))

    if build_mode == 'package':
        # Get build packages
        sdc_packages = get_sdc_build_packages(output_folder)
        for package in sdc_packages:
            if '.tar.bz2' in package:
                # Start test for conda package
                format_print(f'Run tests for sdc conda package: {package}')
                create_conda_env(conda_activate, test_env, python, sdc_env['test'])
                run_command(f'{test_env_activate} && conda install -y {package}')
                run_smoke_tests(sdc_src, test_env_activate)
            elif '.whl' in package:
                # Start test for wheel package
                format_print(f'Run tests for sdc wheel package: {package}')
                create_conda_env(conda_activate, test_env, python, sdc_env['test'] + ['pip'])
                run_command(f'{test_env_activate} && pip install {package}')
                run_smoke_tests(sdc_src, test_env_activate)
    else:
        format_print('Run tests for installed sdc package')
        run_smoke_tests(sdc_src, develop_env_activate)

    format_print('Smoke tests are PASSED')
