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


import pandas
import logging
from datetime import datetime
from module_info import get_submodules_of, print_modules_classes_methods_attributes, ENABLE_LOGGING

# -- Debug logging --------------------------------------------------------------------------------------------------
log_file_name = '../build/pandas_info.log'

# -- Submodules, classes, and methods to be excluded from API Reference ---------------------------------------------
exclude_modules = [
    'pandas.compat',  # This is PRIVATE submodule
    'pandas.util',    # This is PRIVATE submodule
    'pandas.api.extensions',  # This is extension for library developers extending Pandas. Not current interest to SDC
    'pandas.testing',   # Utility functions for testing. Not a priority for SDC
    'pandas.plotting',  # Plotting functions. Not a priority for compiling with SDC
    'pandas.errors',  # Error handling functionality. Not a priority for SDC
    'pandas.api.types',  # Not a priority for SDC
    'pandas.io.formats.style',  # Helps to style dataframes with HTML and CSS. Not a priority for SDC
    'pandas.arrays',  # Array extensions for Numpy. We do not explicitly cover in SDC documentation now
    'pandas.tseries',  # SDC does not yet support Time Series objects
    'pandas.core.dtypes.dtypes',
]

exclude_classes = [
]

exclude_methods = [
]

exclude_attributes = [
]

exclude_functions = [
]


# -- Implements custom skip functions for the parser ----------------------------------------------------------------
def _skip_pandas_module(mod, mod_name):
    for excl_mname in exclude_modules:
        if mod_name.startswith(excl_mname):
            return True
    return not mod_name.startswith('pandas')


def _skip_pandas_class(cls, cls_name):
    return cls_name in exclude_classes


def _skip_pandas_method(cls, method_name):
    # Exclude the method if in the exclude_methods list
    if method_name in exclude_methods:  # Explicit exclusion of the method
        return True

    #  Exclude the method without docstring
    try:
        doc = getattr(cls, method_name).__doc__
        return len(doc) < 1
    except AttributeError:
        return True
    except TypeError:
        return True


def _skip_pandas_function(func, function_name):
    # Exclude the function if in the exclude_functions list
    if function_name in exclude_functions:  # Explicit exclusion of the method
        return True

    #  Exclude the function without docstring
    try:
        doc = func.__doc__
        return len(doc) < 1
    except AttributeError:
        return True
    except TypeError:
        return True


def _skip_pandas_attribute(cls, attr_name):
    # Exclude the attribute if in the exclude_methods list
    if attr_name in exclude_attributes:  # Explicit exclusion of the attribute
        return True

    #  Exclude the attribute without docstring
    try:
        doc = getattr(cls, attr_name).__doc__
        return len(doc) < 1
    except AttributeError:
        return True
    except TypeError:
        return True


def get_pandas_modules():
    inspected_modules = set()
    modules = []
    get_submodules_of(pandas, inspected_modules, modules, _skip_pandas_module, _skip_pandas_class,
                      _skip_pandas_method, _skip_pandas_attribute, _skip_pandas_function)
    return modules


def init_pandas_logging():
    if ENABLE_LOGGING:
        logging.basicConfig(filename=log_file_name, level=logging.DEBUG)
        logging.debug('****************** STARTING THE LOG *************************')
        logging.debug(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))


if __name__ == "__main__":
    # Initialize logging
    init_pandas_logging()

    # Execute parser for Pandas
    modules = get_pandas_modules()

    # You may uncomment this line in case you want to print out generated methods and attributes
    print_modules_classes_methods_attributes(modules)
