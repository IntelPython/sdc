from setuptools import setup, Extension
import platform, os

def readme():
    with open('README.rst') as f:
        return f.read()


_has_h5py = False
HDF5_DIR = ""

if 'HDF5_DIR' in os.environ:
    _has_h5py = True
    HDF5_DIR = os.environ['HDF5_DIR']

# package environment variable is PREFIX during build time
if 'CONDA_BUILD' in os.environ:
    PREFIX_DIR = os.environ['PREFIX']
else:
    PREFIX_DIR = os.environ['CONDA_PREFIX']

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

MPI_LIBS = ['mpi']
H5_COMPILE_FLAGS = []
if platform.system() == 'Windows':
    # use Intel MPI on Windows
    MPI_LIBS = ['impi', 'impicxx']
    # hdf5-parallel Windows build uses CMake which needs this flag
    H5_COMPILE_FLAGS = ['-DH5_BUILT_AS_DYNAMIC_LIB']


ext_io = Extension(name="hio",
                             libraries = ['hdf5'] + MPI_LIBS,
                             include_dirs = [HDF5_DIR+'/include/', PREFIX_DIR+'/include/'],
                             library_dirs = [HDF5_DIR+'/lib/'],
                             extra_compile_args = H5_COMPILE_FLAGS,
                             sources=["hpat/_io.cpp"]
                             )

ext_hdist = Extension(name="hdist",
                             libraries = MPI_LIBS,
                             sources=["hpat/_distributed.cpp"],
                             include_dirs=[PREFIX_DIR+'/include/'],
                             )

ext_chiframes = Extension(name="chiframes",
                             libraries = MPI_LIBS,
                             sources=["hpat/_hiframes.cpp"],
                             include_dirs=[PREFIX_DIR+'/include/'],
                             )


ext_dict = Extension(name="hdict_ext",
                             sources=["hpat/_dict_ext.cpp"]
                             )

ext_str = Extension(name="hstr_ext",
                             sources=["hpat/_str_ext.cpp"],
                             extra_compile_args=['-std=c++11'],
                             extra_link_args=['-std=c++11'],
                             )

ext_quantile = Extension(name="quantile_alg",
                             libraries = MPI_LIBS,
                             sources=["hpat/_quantile_alg.cpp"],
                             include_dirs=[PREFIX_DIR+'/include/'],
                             extra_compile_args=['-std=c++11'],
                             extra_link_args=['-std=c++11'],
                             )

ext_parquet = Extension(name="parquet_cpp",
                             libraries = ['mpi', 'boost_filesystem',
                                            'hpat_parquet_reader'],
                             sources=["hpat/_parquet.cpp"],
                             include_dirs=[PREFIX_DIR+'/include/'],
                             extra_compile_args=['-std=c++11'],
                             extra_link_args=['-std=c++11'],
                             )

ext_daal_wrapper = Extension(name="daal_wrapper",
                             include_dirs = [DAALROOT+'/include'],
                             libraries = ['daal_core', 'daal_thread']+MPI_LIBS,
                             sources=["hpat/_daal.cpp"]
                             )

ext_ros = Extension(name="ros_cpp",
                             include_dirs = ['/opt/ros/lunar/include', '/opt/ros/lunar/include/xmlrpcpp', PREFIX_DIR+'/include/', './ros_include'],
                             extra_link_args='-rdynamic /opt/ros/lunar/lib/librosbag.so /opt/ros/lunar/lib/librosbag_storage.so -lboost_program_options /opt/ros/lunar/lib/libroslz4.so /opt/ros/lunar/lib/libtopic_tools.so /opt/ros/lunar/lib/libroscpp.so -lboost_filesystem -lboost_signals /opt/ros/lunar/lib/librosconsole.so /opt/ros/lunar/lib/librosconsole_log4cxx.so /opt/ros/lunar/lib/librosconsole_backend_interface.so -lboost_regex /opt/ros/lunar/lib/libroscpp_serialization.so /opt/ros/lunar/lib/librostime.so /opt/ros/lunar/lib/libxmlrpcpp.so /opt/ros/lunar/lib/libcpp_common.so -lboost_system -lboost_thread -lboost_chrono -lboost_date_time -lboost_atomic -lpthread -Wl,-rpath,/opt/ros/lunar/lib'.split(),
                             sources=["hpat/_ros.cpp"]
                             )

_ext_mods = [ext_hdist, ext_chiframes, ext_dict, ext_str, ext_quantile]

if _has_h5py:
    _ext_mods.append(ext_io)
if _has_pyarrow:
    _ext_mods.append(ext_parquet)
if _has_daal:
    _ext_mods.append(ext_daal_wrapper)
if _has_ros:
    _ext_mods.append(ext_ros)

setup(name='hpat',
      version='0.1.0',
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
      author='Ehsan Totoni',
      author_email='ehsan.totoni@intel.com',
      packages=['hpat'],
      install_requires=['numba'],
      extras_require={'HDF5': ["h5py"], 'Parquet': ["pyarrow"]},
      ext_modules = _ext_mods)
