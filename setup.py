from setuptools import setup, Extension

def readme():
    with open('README.rst') as f:
        return f.read()

try:
    import h5py
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


ext_io = Extension(name="hio",
                             extra_link_args=['-lmpi','-lhdf5'],
                             sources=["hpat/_io.cpp"]
                             )

ext_hdist = Extension(name="hdist",
                             sources=["hpat/_distributed.cpp"]
                             )

ext_dict = Extension(name="hdict_ext",
                             sources=["hpat/_dict_ext.cpp"]
                             )

ext_str = Extension(name="hstr_ext",
                             sources=["hpat/_str_ext.cpp"]
                             )

ext_parquet = Extension(name="parquet_cpp",
                             extra_link_args=['-lparquet', '-larrow'],
                             sources=["hpat/_parquet.cpp"]
                             )

_ext_mods = [ext_hdist, ext_dict, ext_str]

if _has_h5py:
    _ext_mods.append(ext_io)
if _has_pyarrow:
    _ext_mods.append(ext_parquet)

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
