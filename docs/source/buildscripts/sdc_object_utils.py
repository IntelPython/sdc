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

from inspect import getmembers, ismodule, isclass, isfunction
import sys
import pandas
import sdc
from sdc_doc_utils import is_sdc_user_guide_header, get_docstring, split_title, extract_pandas_name_from

# -- Pandas submodules to be excluded from API Reference ---------------------------------------------
exclude_pandas_submodules = [
    'pandas.compat',  # This is PRIVATE submodule
    'pandas.util',    # This is PRIVATE submodule
    'pandas.api.extensions',  # This is extension for library developers extending Pandas
    'pandas.testing',   # Utility functions for testing. Not a priority for SDC
    'pandas.plotting',  # Plotting functions. Not a priority for compiling with SDC
    'pandas.errors',  # Error handling functionality. Not a priority for SDC
    'pandas.api.types',  # Not a priority for SDC
    'pandas.io.formats.style',  # Helps to style dataframes with HTML and CSS. Not a priority for SDC
    'pandas.arrays',  # Array extensions for Numpy. We do not explicitly cover in SDC documentation now
    'pandas.tseries',  # SDC does not yet support Time Series objects
    'pandas.core.dtypes.dtypes',
]

# -- Intel SDC submodules to be excluded from API Reference -------------------------------------------
exclude_sdc_submodules = [
    'sdc.chiframes',
    'sdc.compiler',
    'sdc.config',
    'sdc.io.pio',
    'sdc.io.pio_api',
    'sdc.io.pio_lower',
    'sdc.utils',
    'sdc.hstr_ext',
    'sdc.datatypes.common_functions',
    'sdc.datatypes.hpat_pandas_dataframe_pass',
    'sdc.decorators',
    'sdc.dict_ext',
    'sdc.hdict_ext',
    'sdc.distributed',
    'sdc.distributed_api',
    'sdc.transport_seq',
    'sdc.distributed_lower',
    'sdc.hdist',
    'sdc.distributed_analysis',
    'sdc.hdatetime_ext',
    'sdc.hiframes',
    'sdc.io.csv_ext',
    'sdc.hio',
    'sdc.hiframes.join',
    'sdc.io.parquet_pio',
    'sdc.parquet_cpp',
    'sdc.shuffle_utils',
    'sdc.str_arr_ext',
    'sdc.str_ext',
    'sdc.timsort',
]

pandas_modules = dict()  # Dictionary of pandas submodules and their classes and functions
sdc_modules = dict()  # Dictionary of Intel SDC submodules and their classes and functions
pandas_sdc_dict = dict()  # Dictionary {<pandas_obj>: <sdc_obj>} that maps Pandas API to respective Intel SDC API


def get_sdc_object(pandas_obj):
    """
    Returns corresponding Intel SDC object for a given Pandas object pandas_obj.

    :param pandas_obj: Pandas object to be matched with Intel SDC object
    :return: Intel SDC object corresponding to pandas_obj
    """
    if pandas_obj in pandas_sdc_dict:
        return pandas_sdc_dict[pandas_obj]
    else:
        return None  # There is no match in Intel SDC to pandas_obj


def get_sdc_object_by_pandas_name(pandas_name):
    """
    Returns corresponding Intel SDC object for a given Pandas object given as string ``pandas_name``.

    This function is needed because :func:`get_sdc_object` cannot uniquely match Intel SDC and Pandas objects.
    For example, the same Pandas object represents :meth:`Series.get` and :meth:`DataFrame.get` methods. As a result
    that :func:`get_sdc_object` will return **some** SDC object that matches respective Pandas object. If you need
    unique match between Pandas and Intel SDC use :func:`get_sdc_object_by_pandas_name` function instead. (Which
    should be the case for majority usecases).

    :param pandas_name: Pandas object to be matched with Intel SDC object
    :return: Intel SDC object corresponding to Pandas object having ``pandas_name`` name
    """
    if pandas_name in pandas_sdc_dict:
        return pandas_sdc_dict[pandas_name]
    else:
        return None  # There is no match in Intel SDC to pandas_obj


def init_pandas_sdc_dict():
    """
    Initializes global dictionary that performs mapping between Pandas objects and SDC objects.

    To function correctly this function must be called after initialization of ``sdc_modules`` and ``pandas_modules``
    lists by :func:`init_sdc_structure` and :func:`init_pandas_structure`` functions respectively.
    """

    def _map_sdc_to_pandas(sdc_obj):
        if isfunction(sdc_obj):
            doc = get_docstring(sdc_obj)

            # The very first section of Intel SDC documentation is expected to start with
            # the User Guide header followed by the name of respective Pandas API.
            # The following code extracts respective Pandas API
            title, text = split_title(doc)
            if is_sdc_user_guide_header((title, text)):
                pandas_name = extract_pandas_name_from(text)
                pandas_obj = get_obj(pandas_name)
                pandas_sdc_dict[pandas_obj] = sdc_obj
                pandas_sdc_dict[pandas_name] = sdc_obj
        return False

    global pandas_sdc_dict
    pandas_sdc_dict = {}

    traverse(sdc_modules, _map_sdc_to_pandas, True)


def get_obj(obj_name):
    """
    Retrieves object corresponding to fully qualified name obj_name.

    The fully qualified name starts with the imported module name visible by sys.modules followed by
    submodules and then classes and finally by class attributes
    :param obj_name: Fully qualified object name string
    :return: If found, returns the object corresponding to obj_name. Otherwise raises exception
    :raises AttributeError: If submodule or attribute does not exists
    """
    split_name = obj_name.split('.')
    split_obj = sys.modules[split_name[0]]

    # Iterate through submodules
    while ismodule(split_obj) and len(split_name) > 1:
        split_name.pop(0)
        not_found = True
        for (name, obj) in getmembers(split_obj):  # Go through members of split_obj
            if split_name[0] == name:
                not_found = False
                break

        if not_found:
            raise AttributeError('Member `' + split_name[0] + '` for `' + obj_name + '` does not exists')

        split_obj = obj

    split_name.pop(0)
    for name in split_name:
        split_obj = getattr(split_obj, name)

    return split_obj


def get_class_methods(cls):
    """
    Returns the list of class methods, accessible by both names and as objects.

    Function ignores internal methods starting with ``_``.

    :param cls: The class object
    :return: List of class methods, each item is the tuple ``(method_name, method_object)``
    """
    return [(func, getattr(cls, func)) for func in dir(cls)
            if callable(getattr(cls, func)) and not func.startswith('_')]


def get_class_attributes(cls):
    """
    Returns the list of class attributes, accessible by both names and as objects.

    Function ignores internal attributes starting with ``_``.

    :param cls: The class object
    :return: List of class attributes, each item is the tuple ``(attribute_name, attribute_object)``
    """
    return [(func, getattr(cls, func)) for func in dir(cls)
            if not callable(getattr(cls, func)) and not func.startswith('_')]


def get_fully_qualified_name(cls):
    """
    Returns fully qualified name of the class.

    :param cls: The class object
    :return: String, fully qualified name
    """
    return repr(cls)[8:-2]


def init_module_structure(module_obj, the_module, inspected, skip_test):
    """
    Initializes hierarchical structure ``the_module``.

    :param module_obj: Module object being traversed.
    :param the_module: Dictionary ``{'module_obj': module_obj, 'submodules': submodules,
        'classes': classes, 'functions': functions}``. The ``submodules`` is the list of
        submodules that belong to ``module_obj``. Each submodule has the same structure as ``the_module``.
        The ``classes`` is the list of classes that belong to ``module_obj``.
        The functions is the list of functions that belong ``to module_obj``.
    :param inspected: Set of already traversed module objects. This set is needed to avoid circular traversal of
        the same module, which may be returned by by ``getmembers`` function multiple times.
    :param skip_test: Function that takes module object as an argument and returns True if this object
        needs to be included in the module structure hierarchy or skipped if False. This function is used as
        a mechanism to customize the structure of modules, classes, and functions. This in turn minimizes following
        structure traversal costs.
    """

    # Returns True if the mod module needs to be ignored
    def _is_skip_module(mod):
        mod_name = mod.__name__
        return '._' in mod_name or mod_name.startswith('_')

    # Returns True if the class cls needs to be ignored
    def _is_skip_class(cls):
        class_name = get_fully_qualified_name(cls)
        return '._' in class_name

    # Returns True if the object obj needs to be ignored
    def _is_internal(obj):
        obj_name = obj.__name__
        return obj_name.startswith('_')

    # ************  The init_module_structure implementation starts here  *******************************************
    if _is_skip_module(module_obj) or module_obj in inspected or skip_test(module_obj):
        return

    inspected.add(module_obj)

    # Traverse submodules, classes, and functions
    submodules = []
    classes = []
    functions = []
    for (name, obj) in getmembers(module_obj):  # Iterate through members of the submodule
        if skip_test(obj):
            continue  # Customizable test for skipping objects as needed

        if ismodule(obj) and obj not in inspected and not _is_skip_module(obj):
            the_submodule = dict()
            init_module_structure(obj, the_submodule, inspected, skip_test)
            submodules.append(the_submodule)

        if isclass(obj) and not _is_skip_class(obj):
            classes.append(obj)

        if isfunction(obj) and not _is_internal(obj):
            functions.append(obj)

    the_module['module_obj'] = module_obj
    the_module['submodules'] = submodules
    the_module['classes'] = classes
    the_module['functions'] = functions


def _print_module(the_module, print_submodules_flag=True):
    """
    Recursively prints ``the_module`` content. Internal utility function for debugging purposes

    :param the_module: Dictionary ``{'module_obj': module_obj, 'submodules': submodules,
        'classes': classes, 'functions': functions}``. The ``submodules`` is the list of
        submodules that belong to ``module_obj``. Each submodule has the same structure as ``the_module``.
        The ``classes`` is the list of classes that belong to ``module_obj``.
        The functions is the list of functions that belong ``to module_obj``.
    """
    print(the_module['module_obj'].__name__)

    print('  CLASSES:')
    for the_class in the_module['classes']:
        print('  - ' + the_class.__name__)

    print('  FUNCTIONS:')
    for the_func in the_module['functions']:
        print('  - ' + the_func.__name__)

    if print_submodules_flag:
        print('  SUBMODULES:')
        for submodule in the_module['submodules']:
            _print_module(submodule, print_submodules_flag)


def traverse(the_module, do_action, traverse_submodules_flag=True):
    """
    Traverses ``the_module`` and performs action :func:`do_action` on each of the objects of the structure.

    :param the_module: Dictionary ``{'module_obj': module_obj, 'submodules': submodules,
        'classes': classes, 'functions': functions}``. The ``submodules`` is the list of
        submodules that belong to ``module_obj``. Each submodule has the same structure as ``the_module``.
        The ``classes`` is the list of classes that belong to ``module_obj``.
        The functions is the list of functions that belong to ``module_obj``.
    :param do_action: Function that takes one parameter ``module_obj`` as input. It returns ``True`` if
        traversal needs to be stopped.
    :param traverse_submodules_flag: True if function must recursively traverse submodules too
    :return: Returns tuple ``(the_module, obj)`` where ``obj`` is the object identified by :func:`do_action` and
        ``the_module`` is the corresponding dictionary structure to which the object belongs. It returns ``None``
        if no object has been identified by the :func:`do_action`
    """
    if do_action(the_module['module_obj']):
        return the_module, the_module['module_obj']

    # Traverse classes of the_module
    for the_class in the_module['classes']:
        if do_action(the_class):
            return the_module, the_class

    # Traverse functions of the_module
    for the_func in the_module['functions']:
        if do_action(the_func):
            return the_module, the_func

    # Recursively traverse submodules of the_module
    if traverse_submodules_flag:
        for submodule in the_module['submodules']:
            the_tuple = traverse(submodule, do_action, traverse_submodules_flag)
            if the_tuple is not None:
                return the_tuple

    return None


def get_pandas_module_structure(pandas_obj):
    """
    Returns corresponding ``the_module`` dictionary structure to which ``pandas_obj`` belongs to.

    This function is typically used in conjunction with :func:`traverse`

    :param pandas_obj:
    :return: ``the_module`` dictionary structure
    """

    def _find(obj):
        return obj == pandas_obj

    the_module, the_object = traverse(pandas_modules, _find)
    return the_module


def init_pandas_structure():
    """
    Initializes ``pandas_modules`` global dictionary representing the structure of Pandas.
    """

    # Test that allows to ignore certain Pandas submodules, classes, or attributes
    def _skip_pandas_test(obj):
        if ismodule(obj):
            name = obj.__name__
            for mod_name in exclude_pandas_submodules:
                if name.startswith(mod_name):
                    return True
            return not name.startswith('pandas')

    global pandas_modules
    pandas_modules = dict()
    inspected_mods = set()
    init_module_structure(pandas, pandas_modules, inspected_mods, _skip_pandas_test)


def init_sdc_structure():
    """
    Initializes ``sdc_modules`` global dictionary representing the structure of Intel SDC.
    """

    # Test that allows to ignore certain Intel SDC submodules, classes, or attributes
    def _skip_sdc_test(obj):
        if ismodule(obj):
            name = obj.__name__
            for mod_name in exclude_sdc_submodules:
                if name.startswith(mod_name):
                    return True
            return not name.startswith('sdc') and not name.startswith('hpat')

    global sdc_modules
    sdc_modules = dict()
    inspected_mods = set()
    init_module_structure(sdc, sdc_modules, inspected_mods, _skip_sdc_test)


if __name__ == "__main__":
    init_pandas_structure()
    _print_module(pandas_modules)

    init_sdc_structure()
    _print_module(sdc_modules)

    init_pandas_sdc_dict()
    print(pandas_sdc_dict)
