import os

from enum import Enum


class Implementation(Enum):
    native = 'native'
    njit = 'njit'
    hpat = 'hpat'


class BaseIO:
    """Base class for IO benchmarks"""
    fname = None

    def remove(self, f):
        """Remove created files"""
        try:
            os.remove(f)
        except OSError:
            # On Windows, attempting to remove a file that is in use
            # causes an exception to be raised
            pass

    def teardown(self, *args, **kwargs):
        self.remove(self.fname)
