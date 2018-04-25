try:
    import hio
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
    import ros_cpp
except ImportError:
    _has_ros = False
else:
    _has_ros = True

try:
    import cv_wrapper
except ImportError:
    _has_opencv = False
else:
    _has_opencv = True
    import hpat.cv_ext

try:
    import hxe_ext
except ImportError:
    _has_xenon = False
else:
    _has_xenon = True
    import hpat.xenon_ext
