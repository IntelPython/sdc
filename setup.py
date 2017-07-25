from setuptools import setup, Extension

def readme():
    with open('README.rst') as f:
        return f.read()

ext_io = Extension(name="hio",
                             extra_link_args=['-lmpi','-lhdf5'],
                             sources=["hpat/_io.c"]
                             )

ext_hdist = Extension(name="hdist",
                             sources=["hpat/_distributed.c"]
                             )

ext_dict = Extension(name="hdict_ext",
                             sources=["hpat/_dict_ext.cpp"]
                             )

ext_str = Extension(name="hstr_ext",
                             sources=["hpat/_str_ext.cpp"]
                             )

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
      ext_modules = [ext_io, ext_hdist, ext_dict, ext_str])
