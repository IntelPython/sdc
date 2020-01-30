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



import os
import platform
import re
import subprocess
import time
import traceback


"""
Create conda environment with desired python and packages
"""
def create_conda_env(conda_activate, env_name, python, packages=[], channels=''):
    packages_list = ' '.join(packages)

    format_print(f'Setup conda {env_name} environment')
    run_command(f'{conda_activate}conda remove -q -y --name {env_name} --all')
    run_command(f'{conda_activate}conda create -q -y -n {env_name} python={python} {packages_list} {channels}')


"""
Create list of packages required for build and test from conda recipe
"""
def get_sdc_env(conda_activate, sdc_src, sdc_recipe, python, numpy, channels):
    def create_env_list(packages, exclude=''):
        env_list = []
        env_set = set()

        for item in packages:
            package = re.search(r"[\w-]+" , item).group()
            version = ''
            if re.search(r"\d+\.[\d\*]*\.?[\d\*]*", item) and '<=' not in item and '>=' not in item:
                version = '={}'.format(re.search(r"\d+\.[\d\*]*\.?[\d\*]*", item).group())
            if package not in env_set and package not in exclude:
                env_set.add(package)
                env_list.append(f'{package}{version}')
        return env_list

    from ruamel_yaml import YAML

    yaml=YAML()
    sdc_recipe_render = os.path.join(sdc_src, 'sdc_recipe_render.yaml')

    # Create environment with conda-build
    sdc_render_env = 'sdc_render'
    sdc_render_env_activate = get_activate_env_cmd(conda_activate, sdc_render_env)
    format_print('Render sdc build and test environment using conda-build')
    create_conda_env(conda_activate, sdc_render_env, python, ['conda-build'])
    run_command('{} && {}'.format(sdc_render_env_activate,
                                  ' '.join([f'conda render --python={python}',
                                                         f'--numpy={numpy}',
                                                         f'{channels} -f {sdc_recipe_render} {sdc_recipe}'])))

    with open(sdc_recipe_render, 'r') as recipe:
        data = yaml.load(recipe)
        build = data['requirements']['build']
        host = data['requirements']['host']
        run = data['requirements']['run']
        test = data['test']['requires']

    return {'build': create_env_list(build + host + run, 'vs2017_win-64'),
            'test': create_env_list(run + test)}


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
        if os.path.isfile(item_path) and re.search(r'^sdc.*\.tar\.bz2$|^sdc.*\.whl$', item):
            sdc_packages.append(item_path)

    return sdc_packages


"""
Return platform specific activation cmd
"""
def get_activate_env_cmd(conda_activate, env_name):
    if platform.system() == 'Windows':
        return f'{conda_activate}activate {env_name}'
    else:
        return f'{conda_activate}source activate {env_name}'


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
Print format message with timestamp
"""
def format_print(msg, new_block=True):
    if new_block:
        print('='*80, flush=True)
    print(f'{time.strftime("%d/%m/%Y %H:%M:%S")}: {msg}', flush=True)


"""
Execute command
"""
def run_command(command):
    print('='*80,  flush=True)
    print(f'{time.strftime("%d/%m/%Y %H:%M:%S")}: {command}', flush=True)
    print('-'*80,  flush=True)
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
