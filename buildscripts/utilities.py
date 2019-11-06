import os
import platform
import re
import subprocess
import traceback

"""
Create conda environment with desired python and packages
"""
def create_conda_env(conda_activate, env_name, python='3.7', packages=None, channels=''):
    print('='*80)
    print('Setup conda {} environment'.format(env_name), flush=True)
    run_command('{}conda remove -y --name {} --all'.format(conda_activate, env_name))
    run_command('{}conda create -y -n {} python={}'.format(conda_activate, env_name, python))
    if packages:
        if platform.system() == 'Windows':
            run_command('{}activate {} && conda install -y {} {}'.format(conda_activate, env_name,
                                                                         ' '.join(packages), channels))
        else:
            run_command('{}source activate {} && conda install -y {} {}'.format(conda_activate, env_name,
                                                                                ' '.join(packages), channels))


"""
Create list of packages required for build and test from conda recipe
"""
def get_sdc_env(conda_activate, sdc_src, sdc_recipe, python='3.7', numpy='1.16', channels=''):
    build_env = []
    test_env  = []
    build_env_set = set()
    test_env_set  = set()
    sdc_recipe_render = os.path.join(sdc_src, 'sdc_recipe_render.yaml')

    # Create environment with conda-build
    sdc_render_env = 'sdc_render'
    sdc_render_env_activate = get_activate_env_cmd(conda_activate, sdc_render_env)
    print('='*80)
    print('Render sdc build and test environment using conda-build')
    create_conda_env(conda_activate, sdc_render_env, python, ['conda-build'])
    run_command('{} && {}'.format(sdc_render_env_activate,
                                  ' '.join(['conda render --python={}'.format(python),
                                                         '--numpy={}'.format(numpy),
                                                         '{} -f {} {}'.format(channels,
                                                                              sdc_recipe_render,
                                                                              sdc_recipe)])))

    try:
        with open(sdc_recipe_render, 'r') as recipe:
            section = 'other'
            requirements_started = False

            for line in recipe:
                # Check current recipe section
                if re.search(r"build:|run:|host:|test:|requires:", line):
                    section = re.search(r"build:|run:|host:|test:|requires:", line).group()
                    requirements_started = True
                    continue
                elif ':' in line:
                    requirements_started = False
                    continue

                # Get package with version (for <= or >= version is not set)
                if requirements_started and re.search(r"^\s+- [\w-]+", line):
                    # Get package name
                    package = re.search(r"^\s+- ([\w-]+)", line).group(1)

                    # Get package version
                    package_version = None
                    if re.search(r"\d+\.[\d\*]*\.?[\d\*]*", line) and '<=' not in line and '>=' not in line:
                        package_version = re.search(r"\d+\.[\d\*]*\.?[\d\*]*", line).group()

                    # Finally add package to build or test environment
                    if section in ['build:', 'host:', 'run:'] and package not in build_env_set:
                        build_env.append('{}{}'.format(package, '=' + package_version if package_version else ''))
                        build_env_set.add(package)
                    if section in ['run:', 'requires:'] and package not in test_env_set:
                        test_env.append('{}{}'.format(package, '=' + package_version if package_version else ''))
                        test_env_set.add(package)
    except:
        print('='*80)
        print('WARNING: Render environment for sdc from {} recipe failed'.format(sdc_recipe))
        print(traceback.format_exc())

    return {'build': build_env, 'test': test_env}


"""
Return list of conda and wheel packages in build_output folder
"""
def get_sdc_build_packages(build_output):
    if platform.system() == 'Windows':
        os_dir = 'win-64'
    elif platform.system() == 'Linux':
        os_dir = 'linux-64'
    elif platform.system() == 'Darwin':
        os_dir = 'osx-64'

    sdc_packages = []
    sdc_build_dir = os.path.join(build_output, os_dir)
    for item in os.listdir(sdc_build_dir):
        item_path = os.path.join(sdc_build_dir, item)
        if os.path.isfile(item_path) and re.search(r'^hpat.*\.tar\.bz2$|^hpat.*\.whl$', item):
            sdc_packages.append(item_path)

    return sdc_packages


"""
Return platform specific activation cmd
"""
def get_activate_env_cmd(conda_activate, env_name):
    if platform.system() == 'Windows':
        return '{}activate {}'.format(conda_activate, env_name)
    else:
        return '{}source activate {}'.format(conda_activate, env_name)


"""
Return platform specific conda activation cmd
"""
def get_conda_activate_cmd(conda_prefix):
    if 'CONDA_PREFIX' in os.environ:
        return ''
    else:
        if platform.system() == 'Windows':
            return '{} && '.format(os.path.join(conda_prefix, 'Scripts', 'activate.bat'))
        else:
            return 'source {} && '.format(os.path.join(conda_prefix, 'bin', 'activate'))


"""
Execute command
"""
def run_command(command):
    print('='*80,  flush=True)
    print(command, flush=True)
    print('='*80,  flush=True)
    if platform.system() == 'Windows':
        subprocess.check_call(command, stdout=None, stderr=None, shell=True)
    else:
        subprocess.check_call(command, executable='/bin/bash', stdout=None, stderr=None, shell=True)


"""
Set environment variable
"""
def set_environment_variable(key, value):
    if key in os.environ:
        os.environ[key] += os.pathsep + value
    else:
        os.environ[key] = value
