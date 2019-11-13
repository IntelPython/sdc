from inspect import getmembers, ismodule, isfunction, isclass
import pandas

"""
def get_class_methods(the_class, class_only=False, instance_only=False, exclude_internal=True):

    #  Include methods of the_class
    def is_included(tup):
        is_method = isfunction(tup[1])
        if is_method:
            bound_to = tup[1].im_self
            print(bound_to)
            internal = tup[1].im_func.func_name[:2] == '__' and tup[1].im_func.func_name[-2:] == '__'
            if internal and exclude_internal:
                include = False
            else:
                include = (bound_to == the_class and not instance_only) or (bound_to == None and not class_only)
        else:
            include = False
        return include
    return filter(is_included, getmembers(the_class))
"""

def parse_submodules(module):

    methods_list = []

    for obj in getmembers(module):
        if isclass(obj[1]):
            print(obj)
            methods_list.append((obj, get_class_methods(obj)))

        if ismodule(obj[1]):
            print(obj)
            methods_list = methods_list + parse_submodules(obj)

        return methods_list


#lst = parse_submodules(pandas)
#for item in lst:
#    print('Class:', item[0], 'Method:', item[1])

for obj in getmembers(pandas):
    if isclass(obj[1]):
        print(obj)
        methods = dir(obj)
        for item in methods:
            print(obj, '.', item)

