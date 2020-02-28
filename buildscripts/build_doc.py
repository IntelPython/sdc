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


import argparse
import os
import shutil

from pathlib import Path
from utilities import SDC_Build_Utilities


def build_doc(sdc_utils):
    os.chdir(str(sdc_utils.doc_path))

    sdc_utils.log_info('Start documentation build', separate=True)
    sdc_utils.run_command('make html')
    sdc_utils.log_info('Documentation build SUCCESSFUL', separate=True)

    return


def publish_doc(sdc_utils):
    doc_local_build = str(sdc_utils.doc_path / 'build' / 'html')
    doc_repo_build = str(sdc_utils.doc_path / sdc_utils.doc_repo_name / sdc_utils.doc_tag)

    git_email = os.environ['SDC_GIT_EMAIL']
    git_username = os.environ['SDC_GIT_USERNAME']
    git_access_token = os.environ['SDC_GIT_TOKEN']
    git_credentials_file = str(Path.home() / '.git-credentials')
    git_credentials = f'https://{git_access_token}:x-oauth-basic@github.com\n'

    sdc_utils.log_info(f'Start documentation publish to {sdc_utils.doc_repo_link}', separate=True)

    os.chdir(str(sdc_utils.doc_path))
    sdc_utils.run_command(f'git clone {sdc_utils.doc_repo_link}')
    os.chdir(str(sdc_utils.doc_repo_name))

    # Set local git options
    sdc_utils.run_command('git config --local credential.helper store')
    with open(git_credentials_file, "w") as fp:
        fp.write(git_credentials)
    sdc_utils.run_command(f'git config --local user.email "{git_email}"')
    sdc_utils.run_command(f'git config --local user.name "{git_username}"')

    sdc_utils.run_command(f'git checkout {sdc_utils.doc_repo_branch}')
    shutil.rmtree(doc_repo_build)
    shutil.copytree(doc_local_build, doc_repo_build)
    sdc_utils.run_command(f'git add -A {sdc_utils.doc_tag}')
    # Check if there is changes
    output = sdc_utils.get_command_output(f'git commit -m "Updated doc release: {sdc_utils.doc_tag}"')
    if 'nothing to commit, working tree clean' not in output:
        sdc_utils.run_command('git push origin HEAD')
        sdc_utils.log_info('Documentation publish SUCCESSFUL', separate=True)
    else:
        sdc_utils.log_info('No changes in documentation', separate=True)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--python', default='3.7', choices=['3.6', '3.7', '3.8'],
                        help='Python version, default = 3.7')
    parser.add_argument('--sdc-channel', default=None, help='Intel SDC channel')
    parser.add_argument('--publish', action='store_true', help='Publish documentation to sdc-doc')

    args = parser.parse_args()

    sdc_utils = SDC_Build_Utilities(args.python, args.sdc_channel)
    sdc_utils.log_info('Build Intel(R) SDC documentation', separate=True)
    sdc_utils.log_info(sdc_utils.line_double)
    sdc_utils.create_environment(['sphinx', 'sphinxcontrib-programoutput'])
    sdc_utils.install_conda_package(['sdc'])

    build_doc(sdc_utils)
    if args.publish:
        publish_doc(sdc_utils)
