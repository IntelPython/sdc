import argparse
import os
import platform
import subprocess
import sys
import traceback

from utilities import render_sdc_env
from utilities import run_command
from utilities import create_conda_env
from utilities import get_sdc_built_packages


if __name__ == '__main__':
    sdc_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sdc_recipe = os.path.join(sdc_src, 'buildscripts', 'sdc-conda-recipe')
    sdc_meta_file = os.path.join(sdc_recipe, 'meta.yaml')

    os.chdir(sdc_src)

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

    build_env = 'sdc-build-env-py{}-numpy{}'.format(python, numpy)
    test_env = 'sdc-test-env-py{}-numpy{}'.format(python, numpy)

    sdc_env = render_sdc_env(sdc_meta_file, python, numpy)

    if platform.system() == 'Windows':
        test_script = os.path.join(sdc_recipe, 'run_test.bat')
        conda_channels = '-c numba -c conda-forge -c defaults -c intel'

        build_env_activate = 'activate {}'.format(build_env)
        test_env_activate = 'activate {}'.format(test_env)

        os.environ['PATH'] = conda_prefix + os.pathsep + os.path.join(conda_prefix, 'Scripts') + os.pathsep + os.environ['PATH']
    else:
        test_script = os.path.join(sdc_recipe, 'run_test.sh')
        conda_channels = '-c numba -c conda-forge -c defaults'

        build_env_activate = 'source activate {}'.format(build_env)
        test_env_activate = 'source activate {}'.format(test_env)

        os.environ['PATH'] = os.path.join(conda_prefix, 'bin') + os.pathsep + os.environ['PATH']


    if run_coverage == True or test_mode == 'develop':
        print('='*80)
        print('Check that SDC installed in develop mode')
        try:
            run_command('{} && python -c "import hpat"'.format(build_env_activate))
        except:
            print('SDC does not installed in developer mode in {} env. Please use "python build.py --build-mode=develop" to do this'.format(build_env))
            sys.exit(1)

    if run_coverage == True
        print('='*80)
        print('Run coverage')
        os.environ['PYTHONPATH'] = '.'
        try:
            run_command('{} && coverage erase && coverage run -m hpat.runtests && coveralls -v')
        except:
            print('='*80)
            print('Coverage fails')
            print(traceback.format_exc())
        sys.exit(0)

    if test_mode == 'develop':
        print('='*80)
        print('Run tests in develop mode')
        run_command('{} && {}'.format(build_env_activate, test_script))


    sdc_packages = get_sdc_built_packages(build_folder)
    try:
        sdc_conda_pkg = sdc_packages[0]
        sdc_wheel_pkg = sdc_packages[1]
    except:
        print('='*80)
        print('ERROR: Built sdc packages not found in {} folder'.format(build_folder))
        print(traceback.format_exc())
        sys.exit(1)


    if test_mode == 'conda':
        create_conda_env(test_env, python, ['conda-build'], conda_channels)
        test_cmd = '{} && {}'.format(test_env_activate,
                                    ' '.join(['conda build --test',
                                                          ' --override-channels {} {}'.format(conda_channels, sdc_conda_pkg)]))

    if test_mode == 'package':
        if package_type == 'conda':
            print('='*80)
            print('Run tests for sdc conda package: {}'.format(package_type, sdc_conda_pkg))
            create_conda_env(test_env, python, sdc_env['test'], conda_channels)
            run_command('{} && conda install -y {}'.format(sdc_conda_pkg))
        else:
            print('='*80)
            print('Run tests for sdc wheel package: {}'.format(sdc_wheel_pkg))
            create_conda_env(test_env, python, sdc_env['test'].append('pip'), conda_channels)
            run_command('{} && pip install {}'.format(sdc_wheel_pkg))

        run_command('{} && {}'.format(test_env_activate, test_script))
