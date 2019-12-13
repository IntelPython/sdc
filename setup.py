# -*- coding: utf-8 -*-
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

from setuptools import setup, Extension, find_packages, Command
import platform
import os
import sys
from docs.source.buildscripts.sdc_build_doc import SDCBuildDoc


# Note we don't import Numpy at the toplevel, since setup.py
# should be able to run without Numpy for pip to discover the
# build dependencies
import numpy.distutils.misc_util as np_misc
#import copy
import versioneer

# String constants for Intel SDC project configuration
SDC_NAME_STR = 'Intel® Scalable Dataframe Compiler'

# Inject required options for extensions compiled against the Numpy
# C API (include dirs, library dirs etc.)
np_compile_args = np_misc.get_info('npymath')

is_win = platform.system() == 'Windows'


def readme():
    with open('README.rst', encoding='utf-8') as f:
        return f.read()


# package environment variable is PREFIX during build time
if 'CONDA_BUILD' in os.environ:
    PREFIX_DIR = os.environ['PREFIX']
else:
    PREFIX_DIR = os.environ['CONDA_PREFIX']
    # C libraries are in \Library on Windows
    if is_win:
        PREFIX_DIR += r'\Library'


try:
    import pyarrow
except ImportError:
    _has_pyarrow = False
else:
    _has_pyarrow = True

_has_opencv = False
OPENCV_DIR = ""

if 'OPENCV_DIR' in os.environ:
    _has_opencv = True
    OPENCV_DIR = os.environ['OPENCV_DIR'].replace('"', '')
    # TODO: fix opencv link
    # import subprocess
    # p_cvconf = subprocess.run(["pkg-config", "--libs", "--static","opencv"], stdout=subprocess.PIPE)
    # cv_link_args = p_cvconf.stdout.decode().split()

ind = [PREFIX_DIR + '/include', ]
lid = [PREFIX_DIR + '/lib', ]
eca = ['-std=c++11', ]  # '-g', '-O0']
ela = ['-std=c++11', ]

MPI_LIBS = ['mpi']

use_impi = False
if use_impi:
    MPI_ROOT = os.environ['I_MPI_ROOT']
    MPI_INC = MPI_ROOT + '/include64/'
    MPI_LIBDIR = MPI_ROOT + '/lib64/'
    MPI_LIBS = ['mpifort', 'mpi', 'mpigi']
    ind = [PREFIX_DIR + '/include', MPI_INC]
    lid = [PREFIX_DIR + '/lib', MPI_LIBDIR]

if is_win:
    # use Intel MPI on Windows
    MPI_LIBS = ['impi']

io_libs = MPI_LIBS
boost_libs = []

if not is_win:
    boost_libs = ['boost_filesystem', 'boost_system']
    io_libs += boost_libs

ext_io = Extension(name="sdc.hio",
                   sources=["sdc/io/_io.cpp", "sdc/io/_csv.cpp"],
                   depends=["sdc/_hpat_common.h", "sdc/_distributed.h",
                            "sdc/_import_py.h", "sdc/io/_csv.h",
                            "sdc/_datetime_ext.h"],
                   libraries=boost_libs,
                   include_dirs=ind + np_compile_args['include_dirs'],
                   library_dirs=lid,
                   extra_compile_args=eca,
                   extra_link_args=ela,
                   language="c++"
                   )

ext_transport_mpi = Extension(name="sdc.transport_mpi",
                              sources=["sdc/transport/hpat_transport_mpi.cpp"],
                              depends=["sdc/_distributed.h"],
                              libraries=io_libs,
                              include_dirs=ind,
                              library_dirs=lid,
                              extra_compile_args=eca,
                              extra_link_args=ela,
                              language="c++"
                              )

ext_transport_seq = Extension(name="sdc.transport_seq",
                              sources=["sdc/transport/hpat_transport_single_process.cpp"],
                              depends=["sdc/_distributed.h"],
                              include_dirs=ind,
                              library_dirs=lid,
                              extra_compile_args=eca,
                              extra_link_args=ela,
                              language="c++"
                              )

ext_hdist = Extension(name="sdc.hdist",
                      sources=["sdc/_distributed.cpp"],
                      depends=["sdc/_hpat_common.h"],
                      extra_compile_args=eca,
                      extra_link_args=ela,
                      include_dirs=ind,
                      library_dirs=lid,
                      )

ext_chiframes = Extension(name="sdc.chiframes",
                          sources=["sdc/_hiframes.cpp"],
                          depends=["sdc/_hpat_sort.h"],
                          extra_compile_args=eca,
                          extra_link_args=ela,
                          include_dirs=ind,
                          library_dirs=lid,
                          )


ext_dict = Extension(name="sdc.hdict_ext",
                     sources=["sdc/_dict_ext.cpp"],
                     extra_compile_args=eca,
                     extra_link_args=ela,
                     include_dirs=ind,
                     library_dirs=lid,
                     )

ext_set = Extension(name="sdc.hset_ext",
                    sources=["sdc/_set_ext.cpp"],
                    extra_compile_args=eca,
                    extra_link_args=ela,
                    include_dirs=ind,
                    library_dirs=lid,
                    )

str_libs = np_compile_args['libraries']

ext_str = Extension(name="sdc.hstr_ext",
                    sources=["sdc/_str_ext.cpp"],
                    libraries=str_libs,
                    define_macros=np_compile_args['define_macros'] + [('USE_BOOST_REGEX', None)],
                    extra_compile_args=eca,
                    extra_link_args=ela,
                    include_dirs=np_compile_args['include_dirs'] + ind,
                    library_dirs=np_compile_args['library_dirs'] + lid,
                    )

ext_dt = Extension(name="sdc.hdatetime_ext",
                   sources=["sdc/_datetime_ext.cpp"],
                   libraries=np_compile_args['libraries'],
                   define_macros=np_compile_args['define_macros'],
                   extra_compile_args=['-std=c++11'],
                   extra_link_args=['-std=c++11'],
                   include_dirs=np_compile_args['include_dirs'],
                   library_dirs=np_compile_args['library_dirs'],
                   language="c++"
                   )

pq_libs = ['arrow', 'parquet']

# Windows MSVC can't have boost library names on command line
# auto-link magic of boost should be used
if not is_win:
    pq_libs += ['boost_filesystem']

# if is_win:
#     pq_libs += ['arrow', 'parquet']
# else:
#     # seperate parquet reader used due to ABI incompatibility of arrow
#     pq_libs += ['hpat_parquet_reader']

ext_parquet = Extension(name="sdc.parquet_cpp",
                        sources=["sdc/io/_parquet.cpp"],
                        libraries=pq_libs,
                        include_dirs=['.'] + ind,
                        define_macros=[('BUILTIN_PARQUET_READER', None)],
                        extra_compile_args=eca,
                        extra_link_args=ela,
                        library_dirs=lid,
                        )

cv_libs = ['opencv_core', 'opencv_imgproc', 'opencv_imgcodecs', 'opencv_highgui']
# XXX cv lib file name needs version on Windows
if is_win:
    cv_libs = [l + '331' for l in cv_libs]

ext_cv_wrapper = Extension(name="sdc.cv_wrapper",
                           sources=["sdc/_cv.cpp"],
                           include_dirs=[OPENCV_DIR + '/include'] + ind,
                           library_dirs=[os.path.join(OPENCV_DIR, 'lib')] + lid,
                           libraries=cv_libs,
                           #extra_link_args = cv_link_args,
                           language="c++",
                           )

_ext_mods = [ext_hdist, ext_chiframes, ext_dict, ext_set, ext_str, ext_dt, ext_io, ext_transport_mpi, ext_transport_seq]

if _has_pyarrow:
    _ext_mods.append(ext_parquet)

if _has_opencv:
    _ext_mods.append(ext_cv_wrapper)


class style(Command):
    """ Command to check and adjust code style
    Usage:
        To check style: python ./setup.py style
        To fix style: python ./setup.py style -a
    """
    user_options = [
        ('apply', 'a', 'Apply codestyle changes to sources.')
    ]
    description = "Code style check and apply (with -a)"
    boolean_options = []

    _result_marker = "Result:"
    _project_directory_excluded = ['build', '.git']

    _c_formatter = 'clang-format-6.0'
    _c_formatter_install_msg = 'pip install clang'
    _c_formatter_command_line = [_c_formatter, '-style=file']
    _c_file_extensions = ['.h', '.c', '.hpp', '.cpp']

    _py_checker = 'pycodestyle'
    _py_formatter = 'autopep8'
    _py_formatter_install_msg = 'pip install --upgrade autopep8\npip install --upgrade pycodestyle'
    # E265 and W503 excluded because I didn't find a way to apply changes automatically
    _py_checker_command_line = [_py_checker, '--ignore=E265,W503']
    _py_formatter_command_line = [_py_formatter, '--in-place', '--aggressive', '--aggressive']
    _py_file_extensions = ['.py']

    def _get_file_list(self, path, search_extentions):
        """ Return file list to be adjusted or checked

        path - is the project base path
        search_extentions - list of strings with files extension to search recurcivly
        """
        files = []
        exluded_directories_full_path = [os.path.join(path, excluded_dir)
                                         for excluded_dir in self._project_directory_excluded]

        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            # match exclude pattern in current directory
            found = False
            for excluded_dir in exluded_directories_full_path:
                if r.find(excluded_dir) >= 0:
                    found = True

            if found:
                continue

            for file in f:
                filename, extention = os.path.splitext(file)
                if extention in search_extentions:
                    files.append(os.path.join(r, file))

        return files

    def initialize_options(self):
        self.apply = 0

    def finalize_options(self):
        pass

    def run(self):
        root_dir = versioneer.get_root()
        print("Project directory is: %s" % root_dir)

        if self.apply:
            self._c_formatter_command_line += ['-i']
        else:
            self._c_formatter_command_line += ['-output-replacements-xml']

        import subprocess

        bad_style_file_names = []

        # C files handling
        c_files = self._get_file_list(root_dir, self._c_file_extensions)
        try:
            for f in c_files:
                command_output = subprocess.Popen(self._c_formatter_command_line + [f], stdout=subprocess.PIPE)
                command_cout, command_cerr = command_output.communicate()
                if not self.apply:
                    if command_cout.find(b'<replacement ') > 0:
                        bad_style_file_names.append(f)
        except BaseException as original_error:
            print("%s is not installed.\nPlease use: %s" % (self._c_formatter, self._c_formatter_install_msg))
            print("Original error message is:\n", original_error)
            exit(1)

        # Python files handling
        py_files = self._get_file_list(root_dir, self._py_file_extensions)
        try:
            for f in py_files:
                if not self.apply:
                    command_output = subprocess.Popen(
                        self._py_checker_command_line + [f])
                    returncode = command_output.wait()
                    if returncode != 0:
                        bad_style_file_names.append(f)
                else:
                    command_output = subprocess.Popen(
                        self._py_formatter_command_line + [f])
                    command_output.wait()
        except BaseException as original_error:
            print("%s is not installed.\nPlease use: %s" % (self._py_formatter, self._py_formatter_install_msg))
            print("Original error message is:\n", original_error)
            exit(1)

        if bad_style_file_names:
            print("Following files style need to be adjusted:")
            for line in bad_style_file_names:
                print(line)
            print("%s Style check failed" % self._result_marker)
        else:
            print("%s Style check passed" % self._result_marker)


# Custom build commands
#
# These commands extend standard setuptools build procedure
#
sdc_build_commands = versioneer.get_cmdclass()
sdc_build_commands['build_doc'] = SDCBuildDoc
sdc_build_commands.update({'style': style})
sdc_version = versioneer.get_version()
sdc_release = 'Alpha (' + versioneer.get_version() + ')'

setup(name=SDC_NAME_STR,
      version=sdc_version,
      description='Numba* extension for compiling Pandas* operations',
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
      keywords='data analytics distributed Pandas Numba',
      url='https://github.com/IntelPython/sdc',
      author='Intel Corporation',
      packages=find_packages(),
      package_data={'sdc.tests': ['*.bz2'], },
      install_requires=['numba'],
      extras_require={'Parquet': ["pyarrow"], },
      cmdclass=sdc_build_commands,
      ext_modules=_ext_mods,
      entry_points={
          "numba_extensions": [
              "init = sdc:_init_extension",
          ]},
      )
