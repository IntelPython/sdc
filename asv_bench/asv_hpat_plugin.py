import os

from asv import util
from asv.console import log
from asv.plugins.conda import _find_conda, Conda


class HPATConda(Conda):
    tool_name = 'hpat_conda'

    def run_executable(self, executable, args, **kwargs):
        log.debug("Running '{0}' in {1}".format(' '.join(args), self.name))
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
        try:
            conda = _find_conda()
        except IOError as e:
            raise util.UserError(str(e))

        return util.check_output([conda, 'run', '-p', self._path, executable] + args, **kwargs)
