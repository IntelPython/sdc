"""
Build HPAT from source

Usage:
python build_hpat.py --env-dir <conda_hpat_env_dir> --build-dir <hpat_source_dir>
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
    env = os.environ.copy()
    if 'I_MPI_ROOT' not in env:
        raise EnvironmentError('I_MPI_ROOT not found in the system environment')

    mpivars_relpath = Path('intel64') / 'bin' / 'mpivars.bat'
    mpi_roots = [Path(mpi_root) for mpi_root in env['I_MPI_ROOT'].split(os.pathsep)]
    mpivars_paths = [mpi_root / mpivars_relpath for mpi_root in mpi_roots]
    existing_mpivars = [mpivars_path for mpivars_path in mpivars_paths if mpivars_path.exists()]

    if not existing_mpivars:
        raise EnvironmentError(f'Could not found neither {", ".join(str(p) for p in mpivars_paths)}')

    first_mpivars_path, *_ = existing_mpivars

    return first_mpivars_path


def _build_win(cwd, env_dir):
    """
    Build HPAT on Windows via the following commands:
        set INCLUDE=%INCLUDE%;%CONDA_PREFIX%\Library\include
        set LIB=%LIB%;%CONDA_PREFIX%\Library\lib
        "%I_MPI_ROOT%"\intel64\bin\mpivars.bat
        set HDF5_DIR=%CONDA_PREFIX%\Library
        python setup.py develop

    :param cwd: current working directory
    :param env_dir: conda environment directory
    """
    logger = logging.getLogger(__name__)
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

    env['HDF5_DIR'] = f'{env_library_dir}'

    mpivars_cmd = [f'{get_mpivars_path()}']
    build_cmd = ['python', 'setup.py', 'develop']

    common_cmd = mpivars_cmd + ['&&'] + build_cmd
    logger.info('Running \'%s\'', subprocess.list2cmdline(common_cmd))
    proc = subprocess.run(common_cmd, cwd=cwd, env=env)
    logger.info(proc.stdout)


def _build_lin(cwd, env_dir):
    """
    Build HPAT on Linux

    :param cwd: current working directory
    :param env_dir: conda environment directory
    """


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
    args = parse_args()
    setup_logging()

    hpat_builder = HPATBuilder(cwd=args.build_dir, env_dir=args.env_dir)
    hpat_builder.build()


if __name__ == "__main__":
    main()
