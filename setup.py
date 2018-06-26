from setuptools import setup, Extension
import platform, os

# Note we don't import Numpy at the toplevel, since setup.py
# should be able to run without Numpy for pip to discover the
# build dependencies
import numpy.distutils.misc_util as np_misc
#import copy

# Inject required options for extensions compiled against the Numpy
# C API (include dirs, library dirs etc.)
np_compile_args = np_misc.get_info('npymath')

is_win = platform.system() == 'Windows'

def readme():
    with open('README.rst') as f:
        return f.read()


_has_h5py = False
HDF5_DIR = ""

if 'HDF5_DIR' in os.environ:
    _has_h5py = True
    HDF5_DIR = os.environ['HDF5_DIR']

#PANDAS_DIR = ""
#if 'PANDAS_DIR' in os.environ:
#    PANDAS_DIR = os.environ['PANDAS_DIR']

# package environment variable is PREFIX during build time
if 'CONDA_BUILD' in os.environ:
    PREFIX_DIR = os.environ['PREFIX']
else:
    PREFIX_DIR = os.environ['CONDA_PREFIX']
    # C libraries are in \Library on Windows
    if is_win:
        PREFIX_DIR += '\Library'


try:
    import pyarrow
except ImportError:
    _has_pyarrow = False
else:
    _has_pyarrow = True

_has_daal = False
DAALROOT = ""

if 'DAALROOT' in os.environ:
    _has_daal = True
    DAALROOT = os.environ['DAALROOT']

_has_ros = False
if 'ROS_PACKAGE_PATH' in os.environ:
    _has_ros = True


_has_opencv = False
OPENCV_DIR = ""

if 'OPENCV_DIR' in os.environ:
    _has_opencv = True
    OPENCV_DIR = os.environ['OPENCV_DIR'].replace('"', '')
    # TODO: fix opencv link
    # import subprocess
    # p_cvconf = subprocess.run(["pkg-config", "--libs", "--static","opencv"], stdout=subprocess.PIPE)
    # cv_link_args = p_cvconf.stdout.decode().split()

_has_xenon = False

if 'HPAT_XE_SUPPORT' in os.environ and  os.environ['HPAT_XE_SUPPORT'] != "0":
    _has_xenon = True

ind = [PREFIX_DIR+'/include',]
lid = [PREFIX_DIR+'/lib',]
eca = ['-std=c++11',]  # '-g', '-O0']
ela = ['-std=c++11',]

MPI_LIBS = ['mpi']
H5_CPP_FLAGS = []

use_impi = False
if use_impi:
    MPI_ROOT = os.environ['I_MPI_ROOT']
    MPI_INC = MPI_ROOT+'/include64/'
    MPI_LIBDIR = MPI_ROOT+'/lib64/'
    MPI_LIBS = [ 'mpifort', 'mpi' , 'mpigi']
    ind = [PREFIX_DIR+'/include', MPI_INC]
    lid = [PREFIX_DIR+'/lib', MPI_LIBDIR]

if is_win:
    # use Intel MPI on Windows
    MPI_LIBS = ['impi', 'impicxx']
    # hdf5-parallel Windows build uses CMake which needs this flag
    H5_CPP_FLAGS = [('H5_BUILT_AS_DYNAMIC_LIB', None)]


ext_io = Extension(name="hio",
                   sources=["hpat/_io.cpp"],
                   libraries = ['hdf5'] + MPI_LIBS + ['boost_filesystem'],
                   include_dirs = [HDF5_DIR+'/include',] + ind,
                   library_dirs = [HDF5_DIR+'/lib',] + lid,
                   define_macros = H5_CPP_FLAGS,
                   extra_compile_args = eca,
                   extra_link_args = ela,
)

ext_hdist = Extension(name="hdist",
                      sources=["hpat/_distributed.cpp"],
                      libraries = MPI_LIBS,
                      extra_compile_args = eca,
                      extra_link_args = ela,
                      include_dirs = ind,
                      library_dirs = lid,
)

ext_chiframes = Extension(name="chiframes",
                          sources=["hpat/_hiframes.cpp"],
                          libraries = MPI_LIBS,
                          depends=["hpat/_hpat_sort.h"],
                          extra_compile_args = eca,
                          extra_link_args = ela,
                          include_dirs = ind,
                          library_dirs = lid,
)


ext_dict = Extension(name="hdict_ext",
                     sources=["hpat/_dict_ext.cpp"],
                     extra_compile_args = eca,
                     extra_link_args = ela,
                     include_dirs = ind,
                     library_dirs = lid,
)

ext_set = Extension(name="hset_ext",
                     sources=["hpat/_set_ext.cpp"],
                     extra_compile_args = eca,
                     extra_link_args = ela,
                     include_dirs = ind,
                     library_dirs = lid,
)

ext_str = Extension(name="hstr_ext",
                    sources=["hpat/_str_ext.cpp"],
                    libraries=['boost_regex'] + np_compile_args['libraries'],
                    define_macros = np_compile_args['define_macros'] + [('USE_BOOST_REGEX', None)],
                    extra_compile_args = eca,
                    extra_link_args = ela,
                    include_dirs = np_compile_args['include_dirs'] + ind,
                    library_dirs = np_compile_args['library_dirs'] + lid,
)

#dt_args = copy.copy(np_compile_args)
#dt_args['include_dirs'] = dt_args['include_dirs'] + [PANDAS_DIR+'/_libs/src/datetime/']
#dt_args['library_dirs'] = dt_args['library_dirs'] + [PANDAS_DIR+'/_libs/tslibs']
#dt_args['libraries'] = dt_args['libraries'] + ['np_datetime']

ext_dt = Extension(name="hdatetime_ext",
                   sources=["hpat/_datetime_ext.cpp"],
                   libraries=np_compile_args['libraries'],
                   define_macros = np_compile_args['define_macros'],
                   extra_compile_args=['-std=c++11'],
                   extra_link_args=['-std=c++11'],
                   include_dirs = np_compile_args['include_dirs'],
                   library_dirs = np_compile_args['library_dirs'],
                   language="c++"
)

ext_quantile = Extension(name="quantile_alg",
                         sources=["hpat/_quantile_alg.cpp"],
                         libraries = MPI_LIBS,
                         extra_compile_args = eca,
                         extra_link_args = ela,
                         include_dirs = ind,
                         library_dirs = lid,
)


# pq_libs = MPI_LIBS + ['boost_filesystem', 'arrow', 'parquet']
pq_libs = MPI_LIBS + ['boost_filesystem']

# if is_win:
#     pq_libs += ['arrow', 'parquet']
# else:
#     # seperate parquet reader used due to ABI incompatibility of arrow
#     pq_libs += ['hpat_parquet_reader']

pq_libs += ['arrow', 'parquet']

ext_parquet = Extension(name="parquet_cpp",
                        sources=["hpat/_parquet.cpp"],
                        libraries = pq_libs,
                        include_dirs = ['.'] + ind,
                        define_macros = [('BUILTIN_PARQUET_READER', None)],
                        extra_compile_args = eca,
                        extra_link_args = ela,
                        library_dirs = lid,
)

#ext_daal_wrapper = Extension(name="daal_wrapper",
#                             include_dirs = [DAALROOT+'/include'],
#                             libraries = ['daal_core', 'daal_thread']+MPI_LIBS,
#                             sources=["hpat/_daal.cpp"]
#                             )

ext_ros = Extension(name="ros_cpp",
                    sources=["hpat/_ros.cpp"],
                    include_dirs = ['/opt/ros/lunar/include', '/opt/ros/lunar/include/xmlrpcpp', PREFIX_DIR+'/include/', './ros_include'],
                    extra_compile_args = eca,
                    extra_link_args = ela  + '-rdynamic /opt/ros/lunar/lib/librosbag.so /opt/ros/lunar/lib/librosbag_storage.so -lboost_program_options /opt/ros/lunar/lib/libroslz4.so /opt/ros/lunar/lib/libtopic_tools.so /opt/ros/lunar/lib/libroscpp.so -lboost_filesystem -lboost_signals /opt/ros/lunar/lib/librosconsole.so /opt/ros/lunar/lib/librosconsole_log4cxx.so /opt/ros/lunar/lib/librosconsole_backend_interface.so -lboost_regex /opt/ros/lunar/lib/libroscpp_serialization.so /opt/ros/lunar/lib/librostime.so /opt/ros/lunar/lib/libxmlrpcpp.so /opt/ros/lunar/lib/libcpp_common.so -lboost_system -lboost_thread -lboost_chrono -lboost_date_time -lboost_atomic -lpthread -Wl,-rpath,/opt/ros/lunar/lib'.split(),
                    library_dirs = lid,
)

cv_libs = ['opencv_core', 'opencv_imgproc', 'opencv_imgcodecs', 'opencv_highgui']
# XXX cv lib file name needs version on Windows
if is_win:
    cv_libs = [l+'331' for l in cv_libs]

ext_cv_wrapper = Extension(name="cv_wrapper",
                           sources=["hpat/_cv.cpp"],
                           include_dirs = [OPENCV_DIR+'/include'] + ind,
                           library_dirs = [os.path.join(OPENCV_DIR,'lib')] + lid,
                           libraries = cv_libs,
                           #extra_link_args = cv_link_args,
                           language="c++",
)

ext_xenon_wrapper = Extension(name="hxe_ext",
                              sources=["hpat/_xe_wrapper.cpp"],
                              #include_dirs = ['/usr/include'],
                              include_dirs = ['.'] + ind,
                              library_dirs = ['.'] + lid,
                              libraries = ['xe'],
                              extra_compile_args = eca,
                              extra_link_args = ela,
)

_ext_mods = [ext_hdist, ext_chiframes, ext_dict, ext_set, ext_str, ext_quantile, ext_dt]

if _has_h5py:
    _ext_mods.append(ext_io)
if _has_pyarrow:
    _ext_mods.append(ext_parquet)
#if _has_daal:
#    _ext_mods.append(ext_daal_wrapper)
if _has_ros:
    _ext_mods.append(ext_ros)
if _has_opencv:
    _ext_mods.append(ext_cv_wrapper)

if _has_xenon:
    _ext_mods.append(ext_xenon_wrapper)

setup(name='hpat',
      version='0.22',
      description='compiling Python code for clusters',
      long_description=readme(),
      classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Compilers",
        "Topic :: System :: Distributed Computing",
      ],
      keywords='data analytics cluster',
      url='https://github.com/IntelLabs/hpat',
      author='Intel',
      packages=['hpat'],
      install_requires=['numba'],
      extras_require={'HDF5': ["h5py"], 'Parquet': ["pyarrow"]},
      ext_modules = _ext_mods)
