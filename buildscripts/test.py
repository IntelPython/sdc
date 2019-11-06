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


if __name__ == '__main__':
    sdc_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sdc_recipe = os.path.join(sdc_src, 'buildscripts', 'sdc-conda-recipe')

    os.chdir(sdc_src)

    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-mode', default='conda',
                        help="""Test mode: 
                        conda:   use conda-build to run tests (default and valid for conda package-type)
                        package: create test environment, install package there and run tests
                        develop: run tests for sdc package already installed in develop mode""")
    parser.add_argument('--package-type', default='conda', help='Package to test: conda or wheel, default = conda')
    parser.add_argument('--python', default='3.7', help='Python version to test with, default = 3.7')
    parser.add_argument('--numpy', default='1.16', help='Numpy version to test with, default = 1.16')
    parser.add_argument('--build-folder', default=os.path.join(sdc_src, 'sdc-build'),
                        help='Built packages location, default = sdc-build')
    parser.add_argument('--conda-prefix', default=None, help='Conda prefix')
    parser.add_argument('--run-coverage', action='store_true', help='Run coverage (sdc must be built in develop mode)')

    args = parser.parse_args()

    test_mode     = args.test_mode
    package_type  = args.package_type
    python        = args.python
    numpy         = args.numpy
    build_folder  = args.build_folder
    conda_prefix  = os.getenv('CONDA_PREFIX', args.conda_prefix)
    run_coverage  = args.run_coverage
    assert conda_prefix is not None, 'CONDA_PREFIX is not defined; Please use --conda-prefix option or activate your conda'

    if package_type == 'wheel' and test_mode == 'conda':
        test_mode = 'package'

    # Init variables
    conda_activate = get_conda_activate_cmd(conda_prefix).replace('"', '')
    test_env = 'sdc-test-env-py{}-numpy{}'.format(python, numpy)
    develop_env = 'sdc-develop-env-py{}-numpy{}'.format(python, numpy)
    test_env_activate = get_activate_env_cmd(conda_activate, test_env)
    develop_env_activate = get_activate_env_cmd(conda_activate, develop_env)

    conda_channels = '-c numba -c conda-forge -c defaults -c intel'

    cmd_delimiter = '&&'
    if platform.system() == 'Windows':
        test_script = os.path.join(sdc_recipe, 'run_test.bat')
        conda_channels = '-c numba -c conda-forge -c defaults -c intel'
        """
        For develop build vs-2015 and vs-2017 runtime is installed.
        If Visual Studio 2017 is not installed, activation returns non-zero code
        thus next command executed with && fails.
        This delimited is used for develop build and after-build smoke tests.
        """
        cmd_delimiter = '&'
    else:
        test_script = os.path.join(sdc_recipe, 'run_test.sh')


    if run_coverage == True:
        print('='*80)
        print('Run coverage')
        print(f'Assume that SDC is installed in develop build-mode to {develop_env} environment')
        print('='*80)
        print('Install scipy and coveralls')
        run_command(f'{develop_env_activate} {cmd_delimiter} conda install -y scipy coveralls {conda_channels}')
        os.environ['PYTHONPATH'] = '.'
        try:
            run_command(f'{develop_env_activate} {cmd_delimiter} coverage erase && coverage run -m hpat.runtests && coveralls -v')
        except:
            print('='*80)
            print('Coverage fails')
            print(traceback.format_exc())
        sys.exit(0)

    if test_mode == 'develop':
        print('='*80)
        print(f'Run tests for sdc installed in {develop_env}')
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
        run_command(f'{develop_env_activate} {cmd_delimiter} {test_script}')

    # Test conda package using conda build
    if test_mode == 'conda':
        create_conda_env(conda_activate, test_env, python, ['conda-build'])
        sdc_packages = get_sdc_build_packages(build_folder)
        for package in sdc_packages:
            if '.tar.bz2' in package:
                run_command(f'{test_env_activate} && conda build --test --override-channels {conda_channels} {package}')

    # Get sdc build and test environment
    sdc_env = get_sdc_env(conda_activate, sdc_src, sdc_recipe, python, numpy, conda_channels)

    # Test specified package type
    if test_mode == 'package':
        print('='*80)
        print(f'Run tests for {package_type} package type')
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
        sdc_packages = get_sdc_build_packages(build_folder)
        for package in sdc_packages:
            if '.tar.bz2' in package and package_type == 'conda':
                print('='*80)
                print(f'Run tests for sdc conda package: {package}')
                create_conda_env(conda_activate, test_env, python, sdc_env['test'], conda_channels)
                run_command(f'{test_env_activate} && conda install -y {package}')
                run_command(f'{test_env_activate} {cmd_delimiter} {test_script}')
            elif '.whl' in package and package_type == 'wheel':
                print('='*80)
                print(f'Run tests for sdc wheel package: {package}')
                create_conda_env(conda_activate, test_env, python, sdc_env['test'] + ['pip'], conda_channels)
                run_command(f'{test_env_activate} && pip install {package}')
                run_command(f'{test_env_activate} {cmd_delimiter} {test_script}')
