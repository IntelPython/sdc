import argparse
import os
import platform
import subprocess
import sys
import traceback

from utilities import create_conda_env
from utilities import get_sdc_env
from utilities import get_sdc_build_packages
from utilities import get_activate_env_cmd
from utilities import get_conda_activate_cmd
from utilities import run_command
from utilities import set_environment_variable


def run_smoke_tests(sdc_src, test_env_activate, cmd_delimiter):
        sdc_pi_example = os.path.join(sdc_src, 'buildscripts', 'sdc_pi_example.py')
        run_command('{} {} python -c "import hpat"'.format(test_env_activate, cmd_delimiter))
        run_command('{} {} python {}'.format(test_env_activate, cmd_delimiter, sdc_pi_example))


if __name__ == '__main__':
    sdc_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sdc_recipe = os.path.join(sdc_src, 'buildscripts', 'sdc-conda-recipe')
    sdc_meta_yaml = os.path.join(sdc_recipe, 'meta.yaml')

    os.chdir(sdc_src)

    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-mode', default='develop',
                        help="""Build mode:
                                develop: install package in source directory (default)
                                install: build and install package in build environment
                                package: build conda and wheel packages""")
    parser.add_argument('--python', default='3.7', help='Python version to build with, default = 3.7')
    parser.add_argument('--numpy', default='1.16', help='Numpy version to build with, default = 1.16')
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
    conda_activate = get_conda_activate_cmd(conda_prefix).replace('"', '')
    build_env = 'sdc-build-env-py{}-numpy{}'.format(python, numpy)
    test_env = 'sdc-test-env-py{}-numpy{}'.format(python, numpy)
    develop_env = 'sdc-develop-env-py{}-numpy{}'.format(python, numpy)
    build_env_activate = get_activate_env_cmd(conda_activate, build_env)
    test_env_activate = get_activate_env_cmd(conda_activate, test_env)
    develop_env_activate = get_activate_env_cmd(conda_activate, develop_env)

    conda_channels = '-c numba -c conda-forge -c defaults -c intel'

    cmd_delimiter = '&&'
    if platform.system() == 'Windows':
        set_environment_variable('INCLUDE', os.path.join('%CONDA_PREFIX%', 'Library', 'include'))
        set_environment_variable('LIB', os.path.join('%CONDA_PREFIX%', 'Library', 'lib'))
        """
        For develop build vs-2015 and vs-2017 runtime is installed.
        If Visual Studio 2017 is not installed, activation returns non-zero code
        thus next command executed with && fails.
        This delimited is used for develop build and after-build smoke tests.
        """
        cmd_delimiter = '&'

    # Get sdc build and test environment
    sdc_env = get_sdc_env(conda_activate, sdc_src, sdc_recipe, python, numpy, conda_channels)

    # Set build command
    if build_mode == 'package':
        create_conda_env(conda_activate, build_env, python, ['conda-build'])
        build_cmd = '{} && {}'.format(build_env_activate,
                                      ' '.join(['conda build --no-test',
                                                            '--python {}'.format(python),
                                                            '--numpy={}'.format(numpy),
                                                            '--output-folder {}'.format(output_folder),
                                                            ' --override-channels {} {}'.format(conda_channels, sdc_recipe)]))
    else:
        create_conda_env(conda_activate, develop_env, python, sdc_env['build'], conda_channels)
        build_cmd = '{} {} python setup.py {}'.format(develop_env_activate, cmd_delimiter, build_mode)

    # Start build
    print('='*80)
    print('Start build')
    run_command(build_cmd)
    print('='*80)
    print('BUILD COMPETED')


    if skip_smoke_tests == True:
        print('='*80)
        print('Smoke tests are skipped due to "--skip-smoke-tests" option')
        sys.exit(0)

    # Start smoke tests
    print('='*80)
    print('Run Smoke tests')
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
                print('='*80)
                print('Run tests for sdc conda package: {}'.format(package))
                create_conda_env(conda_activate, test_env, python, sdc_env['test'], conda_channels)
                run_command('{} && conda install -y {}'.format(test_env_activate, package))
                run_smoke_tests(sdc_src, test_env_activate, cmd_delimiter)
            elif '.whl' in package:
                # Start test for wheel package
                print('='*80)
                print('Run tests for sdc wheel package: {}'.format(package))
                create_conda_env(conda_activate, test_env, python, sdc_env['test'] + ['pip'], conda_channels)
                run_command('{} && pip install {}'.format(test_env_activate, package))
                run_smoke_tests(sdc_src, test_env_activate, cmd_delimiter)
    else:
        print('='*80)
        print('Run tests for sdc package')
        run_smoke_tests(sdc_src, develop_env_activate, cmd_delimiter)

    print('='*80)
    print('Smoke tests are PASSED')
