try:
    from .io import _hdf5
    import h5py
    # TODO: make sure h5py/hdf5 supports parallel
except ImportError:
    _has_h5py = False
else:
    _has_h5py = True

try:
    import pyarrow
except ImportError:
    _has_pyarrow = False
else:
    _has_pyarrow = True

try:
    from . import ros_cpp
except ImportError:
    _has_ros = False
else:
    _has_ros = True

try:
    from . import cv_wrapper
except ImportError:
    _has_opencv = False
else:
    _has_opencv = True
    import hpat.cv_ext

try:
    from . import hxe_ext
except ImportError:
    _has_xenon = False
else:
    _has_xenon = True
    import hpat.io.xenon_ext

config_transport_mpi = True
