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


from inspect import getmembers, ismodule, isclass
import pandas
import logging
from datetime import datetime

# -- Debug logging --------------------------------------------------------------------------------------------------
enable_dubug_logging = True
enable_method_logging = False
enable_class_logging = False
enable_module_logging = True
log_file_name = '../build/pandas_info.log'

HELP_STR_LOG = 'True/False to enable/disable logging (True by default)'
HELP_STR_MODULE_LOG = 'True/False to enable/disable logging of Pandas submodules (True by default)'
HELP_STR_CLASS_LOG = 'True/False to enable/disable logging of classes (True by default)'
HELP_STR_METHOD_LOG = 'True/False to enable/disable logging of methods (True by default)'


# -- Submodules, classes, and methods to be excluded from API Reference ---------------------------------------------
exclude_modules = [
    'pandas.core',    # This is PRIVATE submodule
    'pandas.compat',  # This is PRIVATE submodule
    'pandas.util'     # This is PRIVATE submodule
]

exclude_classes = [
]

exclude_methods = [
]


def _method_logging(s):
    if enable_method_logging:
        logging.debug('[METHOD]' + s)
    return


def _class_logging(s):
    if enable_class_logging:
        logging.debug('[CLASS]' + s)
    return


def _module_logging(s):
    if enable_module_logging:
        logging.debug('[MODULE]' + s)
    return


# -- Returns all classes and respective methods of the module -------------------------------------------------------
def get_submodules_of(module, inspected, module_dict):

    # Returns True if the mod module will not be included in API Reference
    def _skip_module(mod):
        mod_name = mod.__name__  # Get new submodule name
        sk_mod = False

        if mod in inspected:  # Ignore already traversed modules
            sk_mod = True
            _module_logging('`' + mod_name + '` already traversed. Ignoring')
            return sk_mod

        if not mod_name.startswith('pandas'):  # Traverse Pandas submodules only
            sk_mod = True
            _module_logging('`' + mod_name + '` does not start with pandas. Ignoring')
            return sk_mod

        if '._' in mod_name or mod_name.startswith('_'):  # Ignore internal module
            sk_mod = True
            _module_logging('`' + mod_name + '` is internal (starts with _). Ignoring')
            return sk_mod

        for excl_module in exclude_modules:
            if mod_name.startswith(excl_module):  # Ignore submodules in the exclude list
                sk_mod = True
                _module_logging('`' + mod_name + '` is in the exclusion list. Ignoring')
                return sk_mod

        return sk_mod

    # Returns True if the cls class will not be included in API Reference
    def _skip_class(cls):
        sk_class = False
        class_name = repr(cls)[8:-2]  # Get full class name

        if '._' in class_name:  # We are interested only in public classes
            sk_class = True
            _class_logging('`' + class_name + '` is internal. Ignoring')
            return sk_class

        for excl_class in exclude_classes:
            if class_name == excl_class:  # Ignore classes in the exclude list
                sk_class = True
                _class_logging('`' + class_name + '` is in the exclusion list. Ignoring')
                return sk_class

        return sk_class

    # Returns True if the method mname will not be included in API Reference
    def _skip_method(method_name):
        sk_method = False

        if method_name.startswith('_'):  # Ignore internal methods
            sk_method = True
            _method_logging('`' + method_name + '` is internal (starts with __). Ignoring')
            return sk_method

        return sk_method

    # -- get_classes_of() implementation begins
    if _skip_module(module):
        return

    def _generate_class_methods(cls):
        meths = [func for func in dir(cls) if callable(getattr(cls, func)) and not _skip_method(func)]
        for meth in meths:
            _method_logging('Adding method `' + meth + '` to the list')
        return meths

    inspected.add(module)  # Add module is it is not yet traversed
    module_name = module.__name__
    module_dict[module_name] = []

    _module_logging('********************** Inspecting module `' + module_name + '`')

    class_dict = dict()
    # Traverses the mod module classes and submodules
    for (name, obj) in getmembers(module):  # Iterate through members of the submodule
        if isclass(obj):  # We are interested in members, which are classes
            if not _skip_class(obj):
                _class_logging('********************** Inspecting class `' + name + '`')
                methods = _generate_class_methods(obj)  # Inspect methods of the class of interest only
                class_dict[name] = methods
                module_dict[module_name].append(class_dict)

        if ismodule(obj):
            if not _skip_module(obj):
                get_submodules_of(obj, inspected, module_dict)

    return


if __name__ == "__main__":
    import argparse

    # Argument parser
    parser = argparse.ArgumentParser(description='Pandas classes-methods generator')
    parser.add_argument('--log', default=True, help=HELP_STR_LOG, type=bool)
    parser.add_argument('--module_log', default=True, help=HELP_STR_MODULE_LOG, type=bool)
    parser.add_argument('--class_log', default=True, help=HELP_STR_CLASS_LOG, type=bool)
    parser.add_argument('--method_log', default=True, help=HELP_STR_METHOD_LOG, type=bool)

    args = parser.parse_args()
    enable_dubug_logging = args.log
    enable_method_logging = args.method_log
    enable_class_logging = args.class_log
    enable_module_logging = args.module_log

    # Initialize logging
    if enable_dubug_logging:
        logging.basicConfig(filename=log_file_name, level=logging.DEBUG)
        logging.debug('****************** STARTING THE LOG *************************')
        logging.debug(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

    # Execute parser for Pandas
    # Here is the structure of the data being returned
    # class_methods = dict()
    # class_methods['c1'] = ['m1', 'm2', 'm3'] # methods within class c1
    # class_methods['c2'] = ['m4'] # methods within class c2
    # class_methods['c3'] = ['m5'] # methods within class c3
    #
    # module_classes = dict() # dictionary for the list of classes within a given module
    # module_classes['md1'] = []
    # module_classes['md1'].append(class_methods['c1'])
    # module_classes['md1'].append(class_methods['c2'])
    # module_classes['md2'] = []
    # module_classes['md2'].append(class_methods['c3'])
    inspected_modules = set()
    module_classes = dict()
    get_submodules_of(pandas, inspected_modules, module_classes)

    for the_mod, the_classes in module_classes.items():  # module_classes[the_mod] == the_classes
        print(the_mod)
        for the_cls in the_classes:  # the_classes is the list of the_cls items
            for the_cls_name, the_methods in the_cls.items():  # the_cls[the_cls_name] == the_methods
                print(' *', the_cls_name)
                for the_method in the_methods:
                    print('    -', the_method)
