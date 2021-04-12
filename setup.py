# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2017-2020, Intel Corporation All rights reserved.
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
import numba
from docs.source.buildscripts.sdc_build_doc import SDCBuildDoc


# Note we don't import Numpy at the toplevel, since setup.py
# should be able to run without Numpy for pip to discover the
# build dependencies
import numpy.distutils.misc_util as np_misc
import versioneer

# String constants for Intel SDC project configuration
# This name is used for wheel package build
SDC_NAME_STR = 'sdc'

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


# Copypaste from numba
def check_file_at_path(path2file):
    """
    Takes a list as a path, a single glob (*) is permitted as an entry which
    indicates that expansion at this location is required (i.e. version
    might not be known).
    """
    found = None
    path2check = [os.path.split(os.path.split(sys.executable)[0])[0]]
    path2check += [os.getenv(n, '') for n in ['CONDA_PREFIX', 'PREFIX']]
    if sys.platform.startswith('win'):
        path2check += [os.path.join(p, 'Library') for p in path2check]
    for p in path2check:
        if p:
            if '*' in path2file:
                globloc = path2file.index('*')
                searchroot = os.path.join(*path2file[:globloc])
                try:
                    potential_locs = os.listdir(os.path.join(p, searchroot))
                except BaseException:
                    continue
                searchfor = path2file[globloc + 1:]
                for x in potential_locs:
                    potpath = os.path.join(p, searchroot, x, *searchfor)
                    if os.path.isfile(potpath):
                        found = p  # the latest is used
            elif os.path.isfile(os.path.join(p, *path2file)):
                found = p  # the latest is used
    return found


# Search for Intel TBB, first check env var TBBROOT then conda locations
tbb_root = os.getenv('TBBROOT')
if not tbb_root:
    tbb_root = check_file_at_path(['include', 'tbb', 'tbb.h'])

ind = [PREFIX_DIR + '/include', ]
lid = [PREFIX_DIR + '/lib', ]
eca = ['-std=c++11', "-O3", "-DTBB_PREVIEW_WAITING_FOR_WORKERS=1"]  # '-g', '-O0']
ela = ['-std=c++11', ]

io_libs = []

ext_io = Extension(name="sdc.hio",
                   sources=["sdc/io/_io.cpp"],
                   depends=["sdc/_hpat_common.h", "sdc/_distributed.h",
                            "sdc/_import_py.h",
                            "sdc/_datetime_ext.h"],
                   include_dirs=ind + np_compile_args['include_dirs'],
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

ext_set = Extension(name="sdc.hset_ext",
                    sources=["sdc/_set_ext.cpp"],
                    extra_compile_args=eca,
                    extra_link_args=ela,
                    include_dirs=ind,
                    library_dirs=lid,
                    )

ext_sort = Extension(name="sdc.concurrent_sort",
                     sources=[
                        "sdc/native/sort.cpp",
                        "sdc/native/stable_sort.cpp",
                        "sdc/native/module.cpp",
                        "sdc/native/utils.cpp"],
                     extra_compile_args=eca,
                     extra_link_args=ela,
                     libraries=['tbb'],
                     include_dirs=ind + ["sdc/native/", os.path.join(tbb_root, 'include')],
                     library_dirs=lid + [
                        # for Linux
                        os.path.join(tbb_root, 'lib', 'intel64', 'gcc4.4'),
                        # for MacOS
                        os.path.join(tbb_root, 'lib'),
                        # for Windows
                        os.path.join(tbb_root, 'lib', 'intel64', 'vc_mt'),
                     ],
                     language="c++"
                     )

str_libs = np_compile_args['libraries']
numba_include_path = numba.extending.include_path()

ext_str = Extension(name="sdc.hstr_ext",
                    sources=["sdc/_str_ext.cpp"],
                    libraries=str_libs,
                    define_macros=np_compile_args['define_macros'],
                    extra_compile_args=eca,
                    extra_link_args=ela,
                    include_dirs=np_compile_args['include_dirs'] + ind + [numba_include_path],
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

ext_parquet = Extension(name="sdc.parquet_cpp",
                        sources=["sdc/io/_parquet.cpp"],
                        libraries=pq_libs,
                        include_dirs=['.'] + ind,
                        define_macros=[('BUILTIN_PARQUET_READER', None)],
                        extra_compile_args=eca,
                        extra_link_args=ela,
                        library_dirs=lid,
                        )

print("Using Intel TBB from:", tbb_root)
ext_conc_dict = Extension(
    name="sdc.hconc_dict",
    sources=["sdc/native/conc_dict_module.cpp", "sdc/native/utils.cpp"],
    include_dirs=[os.path.join(tbb_root, 'include')] + [numba_include_path, "sdc/native/"],
    libraries=['tbb'],
    library_dirs=[
      # for Linux
      os.path.join(tbb_root, 'lib', 'intel64', 'gcc4.4'),
      # for MacOS
      os.path.join(tbb_root, 'lib'),
      # for Windows
      os.path.join(tbb_root, 'lib', 'intel64', 'vc_mt'),
    ],
    language="c++",
    )

_ext_mods = [ext_hdist, ext_chiframes, ext_set, ext_str, ext_dt, ext_io, ext_transport_seq, ext_sort,
             ext_conc_dict,
]

# Support of Parquet is disabled because HPAT pipeline does not work now
# if _has_pyarrow:
#     _ext_mods.append(ext_parquet)


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
sdc_version = versioneer.get_version().split('+')[0]

setup(name=SDC_NAME_STR,
      version=sdc_version,
      description='Numba* extension for compiling Pandas* operations',
      long_description=readme(),
      long_description_content_type='text/markdown',
      classifiers=[
          "Development Status :: 4 - Beta",
          "Intended Audience :: Developers",
          "Operating System :: POSIX :: Linux",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3.7",
          "Topic :: Software Development :: Compilers",
          "Topic :: System :: Distributed Computing",
      ],
      keywords='data analytics distributed Pandas Numba',
      url='https://github.com/IntelPython/sdc',
      license='BSD',
      author='Intel Corporation',
      maintainer="Intel Corp.",
      maintainer_email="scripting@intel.com",
      platforms=["Windows", "Linux", "Mac OS-X"],
      python_requires='>=3.6',
      packages=find_packages(),
      package_data={'sdc.tests': ['*.bz2'], },
      install_requires=[
          'numpy>=1.16',
          'pandas==1.2.0',
          'pyarrow==2.0.0',
          'numba==0.52.0',
          'tbb'
          ],
      cmdclass=sdc_build_commands,
      ext_modules=_ext_mods,
      entry_points={
          "numba_extensions": [
              "init = sdc:_init_extension",
          ]},
      )
