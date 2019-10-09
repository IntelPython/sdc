import os
import itertools
from shutil import copyfile

cur_dir = os.getcwd()
api_dir = os.path.join(cur_dir, "usersource", "api")
os.chdir(api_dir)
for file in os.listdir(os.getcwd()):
    with open(file, "r") as rstfile:
        l = rstfile.readlines()
    with open(file, 'w') as rstfile:
        rstfile.truncate(0)
        rstfile.writelines(l[3:])