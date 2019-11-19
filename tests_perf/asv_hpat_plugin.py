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
Plugin to run HPAT performance tests under Conda environment via Airspeed Velocity.
"""
import multiprocessing
import os

from asv import util
from asv.console import log
from asv.plugins.conda import Conda, _find_conda


class HPATConda(Conda):
    tool_name = 'hpat_conda'

    @property
    def conda_executable(self):
        """Find conda executable."""
        try:
            return _find_conda()
        except IOError as e:
            raise util.UserError(str(e))

    def activate_conda(self, executable, args):
        """Wrap command with arguments under conda environment"""
        return [self.conda_executable, 'run', '-p', self._path, executable] + args

    def activate_mpi(self, executable, args):
        """Wrap command with arguments under mpiexec"""
        if util.WIN:
            # Use processes number from system variable MPIEXEC_NP or system
            mpiexec_np = os.environ.get('MPIEXEC_NP', str(multiprocessing.cpu_count()))
            mpi_args = ['mpiexec', '-n', mpiexec_np]
        else:
            # mpiexec under util.check_output hangs on Linux, so temporarily disabling MPI on Linux
            mpi_args = []

        return mpi_args + [executable] + args

    def run(self, args, **kwargs):
        log.debug("Running '{0}' in {1}".format(' '.join(args), self.name))
        # mpiexec removes quotes from command line args, so escaping the quotes in the command
        escaped_args = [arg.replace('"', '\\"') for arg in args]
        executable, *args = self.activate_mpi('python', escaped_args)

        return self.run_executable(executable, args, **kwargs)

    def run_executable(self, executable, args, **kwargs):
        env = dict(kwargs.pop('env', os.environ), PYTHONNOUSERSITE='True').copy()
        env.update(self._env_vars)

        # Insert bin dirs to PATH
        if 'PATH' in env:
            paths = env['PATH'].split(os.pathsep)
        else:
            paths = []

        if util.WIN:
            subpaths = [
                'Library\\mingw-w64\\bin',
                'Library\\bin',
                'Library\\usr\\bin',
                'Scripts'
            ]
            for sub in subpaths[::-1]:
                paths.insert(0, os.path.join(self._path, sub))
            paths.insert(0, self._path)
        else:
            paths.insert(0, os.path.join(self._path, 'bin'))

        # Discard PYTHONPATH, which can easily break the environment isolation
        if 'ASV_PYTHONPATH' in env:
            env['PYTHONPATH'] = env['ASV_PYTHONPATH']
            env.pop('ASV_PYTHONPATH', None)
        else:
            env.pop('PYTHONPATH', None)

        # When running pip, we need to set PIP_USER to false, as --user (which
        # may have been set from a pip config file) is incompatible with virtualenvs.
        kwargs['env'] = dict(env, PIP_USER=str('false'), PATH=str(os.pathsep.join(paths)))
        conda_cmd = self.activate_conda(executable, args)

        return util.check_output(conda_cmd, **kwargs)
