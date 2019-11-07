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


"""
    This script requires developers to add the following information:
    1. add file and function name to srcfiles_srcfuncs
    2. add file and directory name to srcdir_srcfiles
    3. add expected display name for the function to display_names
"""


import os
import itertools
from shutil import copyfile

"""
    Add the function names with the src file in this dictionary
    If the file is already present, just add the func name in the respective values
    Create new entry if the srcfile is not present

    srcfiles_srcfuncs = { srcfile : [func1, func2..]}
    srcfile : file containing the function that should be renamed
    [func1, func2..] : list of function names that should be changed

"""
srcfiles_srcfuncs = {
    "hpat_pandas_series_functions.py": ["hpat_pandas_series_append", "hpat_pandas_series_ne", "hpat_pandas_series_iloc"]
}

"""
     Add the filenames and the parent directory in this dictionary
     If the dir is already present in this list, just add the filename in the respective values
     Create a new entry if the dir is not present in this dictionary

     srcdir_srcfiles = { parentdir : [filename1, filename2..]}
     parentdir : Parent directory for the file
     [filename1, filename2 ..] : List of files that have the functions to be renamed

 """
srcdir_srcfiles = {
    "hpat/datatypes": ["hpat_pandas_series_functions.py"],
    "hpat/hiframes": ["aggregate.py", "boxing.py"]
}


# Add the function name that will replace the original name and should be displayed in documentation
# Always add new name at the ends. Do not change the order
display_names = ['append', 'ne', 'iloc']
cur_dir = os.getcwd()


# This is the dir where all the source files will be copied
src_copy_dir = os.path.join(cur_dir, "API_Doc")
if not os.path.exists(src_copy_dir):
    os.mkdir(src_copy_dir)

# Copy all required srcfiles
for dir in srcdir_srcfiles:
    file_list = srcdir_srcfiles[dir]
    for f in file_list:
        src_file = os.path.join(cur_dir, dir, f)
        dst_file = os.path.join(cur_dir, "API_Doc", f)
        copyfile(src_file, dst_file)

os.chdir(src_copy_dir)

# Change the function names in copied files
i = 0
for filename in srcfiles_srcfuncs:
    func_list = srcfiles_srcfuncs[filename]
    with open(filename, 'r') as fn:
        content = fn.read()
        for func in func_list:
            content = content.replace(func, display_names[i])
            i += 1
    with open(filename, 'w') as fn:
        fn.write(content)
