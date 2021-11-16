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


import json
import os
import platform
import re
import subprocess
import time
import traceback

from pathlib import Path

from conda.cli.python_api import Commands as Conda_Commands
from conda.cli.python_api import run_command as exec_conda_command


class SDC_Build_Utilities:
    def __init__(self, python, channels=None, sdc_channel=None):
        self.src_path = Path(__file__).resolve().parent.parent
        self.env_name = 'sdc_env'
        self.python = python
        self.output_folder = self.src_path / 'sdc-build'
        self.recipe = self.src_path / 'conda-recipe'

        self.line_double = '='*80
        self.line_single = '-'*80

        # Set channels
        build_channels = ['-c', 'main', '-c', 'conda-forge', '-c', 'defaults']
        self.channel_list = build_channels if channels is None else channels.split()
        if sdc_channel:
            self.sdc_channel = Path(sdc_channel).resolve().as_uri()
            self.channel_list = ['-c', self.sdc_channel] + self.channel_list
        else:
            self.sdc_channel = 'intel/label/beta'
            # keep SDC channel but do not add it to env channels
        self.channels = ' '.join(self.channel_list)

        # Conda activate command and conda croot (build) folder
        if platform.system() == 'Windows':
            self.croot_folder = 'C:\\cb'
            self.env_activate = f'activate {self.env_name}'
        else:
            self.croot_folder = '/cb'
            self.env_activate = f'source activate {self.env_name}'

        # build_doc vars
        self.doc_path = self.src_path / 'docs'
        self.doc_tag = 'dev'
        self.doc_repo_name = 'sdc-doc'
        self.doc_repo_link = 'https://github.com/IntelPython/sdc-doc.git'
        self.doc_repo_branch = 'gh-pages'

        # run_examples vars
        self.examples_path = self.src_path / 'examples'

    def create_environment(self, packages_list=[]):
        assert type(packages_list) == list, 'Argument should be a list'

        self.log_info(f'Create {self.env_name} conda environment')

        # Clear Intel SDC environment
        remove_args = ['-q', '-y', '--name', self.env_name, '--all']
        self.__run_conda_command(Conda_Commands.REMOVE, remove_args)

        # Create Intel SDC environment
        create_args = ['-q', '-y', '-n', self.env_name, f'python={self.python}']
        create_args += packages_list + self.channel_list + ['--override-channels']
        self.log_info(self.__run_conda_command(Conda_Commands.CREATE, create_args))

        return

    def install_conda_package(self, packages_list, channels=None):
        assert type(packages_list) == list, 'Argument should be a list'

        self.log_info(f'Install {" ".join(packages_list)} to {self.env_name} conda environment')
        install_args = ['-n', self.env_name]
        replace_channels = channels.split() if channels else self.channel_list
        install_args += replace_channels + ['--override-channels', '-q', '-y'] + packages_list
        self.log_info(self.__run_conda_command(Conda_Commands.INSTALL, install_args))

        return

    def install_wheel_package(self, packages_list):
        return

    def __run_conda_command(self, conda_command, command_args):
        self.log_info(f'conda {conda_command} {" ".join(command_args)}')
        output, errors, return_code = exec_conda_command(conda_command, *command_args, use_exception_handler=True)
        if return_code != 0:
            raise Exception(output + errors + f'Return code: {str(return_code)}')
        return output

    def run_command(self, command):
        self.log_info(command)
        self.log_info(self.line_single)
        if platform.system() == 'Windows':
            subprocess.check_call(f'{self.env_activate} && {command}', stdout=None, stderr=None, shell=True)
        else:
            subprocess.check_call(f'{self.env_activate} && {command}', executable='/bin/bash',
                                  stdout=None, stderr=None, shell=True)

    def get_command_output(self, command):
        self.log_info(command)
        self.log_info(self.line_single)
        if platform.system() == 'Windows':
            output = subprocess.check_output(f'{self.env_activate} && {command}', universal_newlines=True, shell=True)
        else:
            output = subprocess.check_output(f'{self.env_activate} && {command}', executable='/bin/bash',
                                             universal_newlines=True, shell=True)
        print(output, flush=True)
        return output

    def log_info(self, msg, separate=False):
        if separate:
            print(f'{time.strftime("%d/%m/%Y %H:%M:%S")}: {self.line_double}', flush=True)
        print(f'{time.strftime("%d/%m/%Y %H:%M:%S")}: {msg}', flush=True)

    def get_sdc_version_from_channel(self):
        python_version = 'py' + self.python.replace('.', '')

        search_args = ['sdc', '-c', self.sdc_channel, '--override-channels', '--json']
        search_result = self.__run_conda_command(Conda_Commands.SEARCH, search_args)

        repo_data = json.loads(search_result)
        for package_data in reversed(repo_data['sdc']):
            sdc_version = package_data['version']
            sdc_build = package_data['build']
            if python_version in sdc_build:
                break

        return f'{sdc_version}={sdc_build}'
