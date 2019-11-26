# *****************************************************************************
# Copyright (c) 2019, Intel Corporation All rights reserved.
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


"""
Build HPAT from source
Usage:
python build_sdc.py --env-dir <conda_hpat_env_dir> --build-dir <hpat_source_dir>
"""
import argparse
import logging
import os
import platform
import subprocess

from pathlib import Path


def setup_logging():
    """Setup logger"""
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    logger.addHandler(stream_handler)

    return logger


def get_mpivars_path():
    """Get path to the mpivars script"""
    env = os.environ.copy()
    if 'I_MPI_ROOT' not in env:
        raise EnvironmentError('I_MPI_ROOT not found in the system environment')

    mpi_roots = [Path(mpi_root) for mpi_root in env['I_MPI_ROOT'].split(os.pathsep)]
    mpivars_paths = [mpi_root / 'intel64' / 'bin' / 'mpivars.bat' for mpi_root in mpi_roots]
    existing_mpivars = [mpivars_path for mpivars_path in mpivars_paths if mpivars_path.exists()]

    if not existing_mpivars:
        raise EnvironmentError(f'Could not found neither {", ".join(str(p) for p in mpivars_paths)}')

    first_mpivars_path, *_ = existing_mpivars

    return first_mpivars_path


def run_cmd(cmd, cwd=None, env=None):
    """
    Run specified command with logging

    :param cmd: command
    :param cwd: current working directory
    :param env: environment
    """
    logger = logging.getLogger(__name__)

    logger.info('Running \'%s\'', subprocess.list2cmdline(cmd))
    proc = subprocess.run(cmd, cwd=cwd, env=env)
    logger.info(proc.stdout)


def _build_win(cwd, env_dir):
    """
    Build HPAT on Windows via the following commands:
        set INCLUDE=%INCLUDE%;%CONDA_PREFIX%\\Library\\include
        set LIB=%LIB%;%CONDA_PREFIX%\\Library\\lib
        "%I_MPI_ROOT%"\\intel64\\bin\\mpivars.bat
        python setup.py develop

    :param cwd: current working directory
    :param env_dir: conda environment directory
    """
    env = os.environ.copy()

    env_library_dir = env_dir / 'Library'
    include_dirs = []
    if 'INCLUDE' in env:
        include_dirs.append(env['INCLUDE'])
    include_dirs.append(f'{env_library_dir / "include"}')
    env['INCLUDE'] = os.pathsep.join(include_dirs)

    lib_dirs = []
    if 'LIB' in env:
        lib_dirs.append(env['LIB'])
    lib_dirs.append(f'{env_library_dir / "lib"}')
    env['LIB'] = os.pathsep.join(lib_dirs)

    mpivars_cmd = [f'{get_mpivars_path()}']
    build_cmd = ['python', 'setup.py', 'develop']
    common_cmd = mpivars_cmd + ['&&'] + build_cmd
    run_cmd(common_cmd, cwd=cwd, env=env)


def _build_lin(cwd, env_dir):
    """
    Build HPAT on Linux via the following commands:
        python setup.py develop

    :param cwd: current working directory
    :param env_dir: conda environment directory
    """
    env = os.environ.copy()

    cmd = ['python', 'setup.py', 'develop']
    run_cmd(cmd, cwd=cwd, env=env)


def get_builder():
    """Get HPAT builder according to system OS"""
    system_platform = platform.system()
    if system_platform == 'Windows':
        return _build_win
    if system_platform == 'Linux':
        return _build_lin

    raise ValueError(f'Unknown OS: {system_platform}')


class HPATBuilder:
    def __init__(self, cwd, env_dir):
        self.cwd = cwd
        self.env_dir = env_dir

    def build(self):
        """Build HPAT via obtained builder"""
        builder = get_builder()
        return builder(self.cwd, self.env_dir)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-dir', required=True, type=Path, help='Path to the currently active environment root')
    parser.add_argument('--build-dir', required=True, type=Path, help='Path to the build directory')

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    setup_logging()

    hpat_builder = HPATBuilder(cwd=args.build_dir, env_dir=args.env_dir)
    hpat_builder.build()


if __name__ == "__main__":
    main()
