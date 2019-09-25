import os
import itertools
from shutil import copyfile

# Add the directory names for the files here
# For example, if the file containing the function is hpat/hiframes/aggregate.py
# then add "hpat/hiframes" in the below list
srcfunc_dir = ["hpat/datatypes"]

#Add the filename containing the function here
srcfile_list = ["hpat_pandas_series_functions.py"]

#Add the original function names here
original_func_names = ['hpat_pandas_series_append', 'hpat_pandas_series_ne']
total_func_num = len(original_func_names)

#Add the function name that should be displayed in documentation
display_names = ['append', 'ne']
cur_dir = os.getcwd()

#this is the dir where all the source files will be copied
temp_src_dir = os.path.join(cur_dir, "..", "API_Doc")

#Copy all the files in a separate dir
for dir, file in zip(srcfunc_dir, srcfile_list):
    src_file = os.path.join(cur_dir, "..", dir, file)
    dst_file = os.path.join(cur_dir, "..", "API_Doc", file)
    copyfile(src_file, dst_file)

os.chdir(temp_src_dir)

#Search for the original function names in eachfile and replace it with the names that should be displayed
for filename in os.listdir(temp_src_dir):
    with open(filename, 'r+') as f:
        data = f.read()
        for i in range(total_func_num):
            data = data.replace(original_func_names[i], display_names[i])
        f.seek(0, 0)
        f.write(data)

            

        

