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

    args = parser.parse_args()

    test_mode     = args.test_mode
    package_type  = args.package_type
    python        = args.python
    numpy         = args.numpy
    build_folder  = args.build_folder
    conda_prefix  = os.getenv('CONDA_PREFIX', args.conda_prefix)
    run_coverage  = args.run_coverage
    assert conda_prefix is not None, 'CONDA_PREFIX is not defined; Please use --conda-prefix option or activate your conda'

    # Init variables
    conda_activate       = get_conda_activate_cmd(conda_prefix).replace('"', '')
    test_env             = f'sdc-test-env-py{python}-numpy{numpy}'
    develop_env          = f'sdc-develop-env-py{python}-numpy{numpy}'
    test_env_activate    = get_activate_env_cmd(conda_activate, test_env)
    develop_env_activate = get_activate_env_cmd(conda_activate, develop_env)

    conda_channels = '-c conda-forge -c numba -c intel -c defaults --override-channels'

    if platform.system() == 'Windows':
        test_script = os.path.join(sdc_recipe, 'run_test.bat')
    else:
        test_script = os.path.join(sdc_recipe, 'run_test.sh')


    if run_coverage == 'True':
        format_print('Run coverage')
        format_print(f'Assume that SDC is installed in develop build-mode to {develop_env} environment', new_block=False)
        format_print('Install scipy and coveralls')
        run_command(f'{develop_env_activate} && conda install -q -y scipy coveralls')
        os.environ['PYTHONPATH'] = '.'
        os.environ['HDF5_DIR'] = conda_prefix
        try:
            run_command(f'{develop_env_activate} && python -m hpat.tests.gen_test_data && coverage erase && coverage run -m hpat.runtests && coveralls -v')
        except:
            format_print('Coverage fails')
            print(traceback.format_exc())
        sys.exit(0)

    if test_mode == 'develop':
        format_print(f'Run tests for sdc installed to {develop_env}')
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
                run_command(f'{test_env_activate} && conda build --test --prefix-length 10 {package}')
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
