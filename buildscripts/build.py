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
    parser.add_argument('--build-mode', default='develop',
                        help='Build mode: develop (default) - install package in source directory; package - build conda and wheel packages')
    parser.add_argument('--python', default='3.7', help='Python version to build with, default = 3.7')
    parser.add_argument('--numpy', default='1.16', help='Numpy version to build with, default = 1.16')
    parser.add_argument('--output-folder', default=os.path.join(sdc_src, 'sdc-build'),
                        help='Output folder for built packages, default = sdc-build')
    parser.add_argument('--conda-prefix', default=None, help='Conda prefix')

    args = parser.parse_args()

    build_mode    = args.build_mode
    python        = args.python
    numpy         = args.numpy
    output_folder = args.output_folder
    conda_prefix  = os.getenv('CONDA_PREFIX', args.conda_prefix)
    assert conda_prefix is not None, 'CONDA_PREFIX is not defined; Please use --conda-prefix option or activate your conda'

    build_env = 'sdc-build-env-py{}-numpy{}'.format(python, numpy)
    test_env = 'sdc-test-env-py{}-numpy{}'.format(python, numpy)

    sdc_env = render_sdc_env(sdc_meta_file, python, numpy)

    if platform.system() == 'Windows':
        conda_channels = '-c numba -c conda-forge -c defaults -c intel'

        mpi_vars = os.path.join(conda_prefix, 'Library', 'bin', 'mpivars.bat')
        build_env_activate = 'activate {}'.format(build_env)
        test_env_activate = 'activate {}'.format(test_env)
        if build_mode == 'develop':
            build_env_activate = '{} && {}'.format(build_env_activate, mpi_vars)

        os.environ['PATH'] = conda_prefix + os.pathsep + os.path.join(conda_prefix, 'Scripts') + os.pathsep + os.environ['PATH']
        os.environ['INCLUDE'] += os.pathsep + os.path.join(conda_prefix, 'Library', 'include')
        os.environ['LIB'] += os.pathsep + os.path.join(conda_prefix, 'Library', 'lib')
    else:
        conda_channels = '-c numba -c conda-forge -c defaults'

        build_env_activate = 'source activate {}'.format(build_env)
        test_env_activate = 'source activate {}'.format(test_env)

        os.environ['PATH'] = os.path.join(conda_prefix, 'bin') + os.pathsep + os.environ['PATH']


    if build_mode == 'develop':
        create_conda_env(build_env, python, sdc_env['build'], conda_channels)
        build_cmd = '{} && python setup.py develop'.format(build_env_activate)
    else:
        create_conda_env(build_env, python, ['conda-build'], conda_channels)
        build_cmd = '{} && {}'.format(build_env_activate,
                               ' '.join(['conda build --no-test',
                                                     '--python {}'.format(python),
                                                     '--numpy={}'.format(numpy),
                                                     '--output-folder {}'.format(output_folder),
                                                     ' --override-channels {} {}'.format(conda_channels, sdc_recipe)]))

    print('='*80)
    print('Start build')
    run_command(build_cmd)

    print('='*80)
    print('Run Smoke tests')
    sdc_packages = get_sdc_built_packages(output_folder)
    try:
        sdc_conda_pkg = sdc_packages[0]
        sdc_wheel_pkg = sdc_packages[1]
    except:
        print('='*80)
        print('ERROR: Built sdc packages not found in {} folder'.format(output_folder))
        print(traceback.format_exc())
        sys.exit(1)

    print('='*80)
    print('Run tests for sdc conda package: {}'.format(sdc_conda_pkg))
    create_conda_env(test_env, python, sdc_env['test'], conda_channels)
    run_command('{} && conda install -y {}'.format(sdc_conda_pkg))
    run_command('{} && python -c "import hpat"'.format(test_env_activate))
    run_command('{} && python sdc_pi_example.py'.format(test_env_activate))

    print('='*80)
    print('Run tests for sdc wheel package: {}'.format(sdc_wheel_pkg))
    create_conda_env(test_env, python, sdc_env['test'].append('pip'), conda_channels)
    run_command('{} && pip install {}'.format(sdc_wheel_pkg))
    run_command('{} && python -c "import hpat"'.format(test_env_activate))
    run_command('{} && python sdc_pi_example.py'.format(test_env_activate))
