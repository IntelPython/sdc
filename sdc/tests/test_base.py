import warnings
import unittest


class TestCase(unittest.TestCase):
    """Base class for all tests"""

    def numba_jit(self, *args, **kwargs):
        import numba
        if 'nopython' in kwargs:
            warnings.warn('nopython is set to True and is ignored', RuntimeWarning)
        if 'parallel' in kwargs:
            warnings.warn('parallel is set to True and is ignored', RuntimeWarning)
        kwargs.update({'nopython': True, 'parallel': True})
        return numba.jit(*args, **kwargs)

    def sdc_jit(self, *args, **kwargs):
        import sdc
        return sdc.jit(*args, **kwargs)

    def jit(self, *args, **kwargs):
        from sdc import config
        if config.config_pipeline_hpat_default:
            return self.sdc_jit(*args, **kwargs)
        else:
            return self.numba_jit(*args, **kwargs)
