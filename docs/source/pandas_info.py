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
from module_info import get_submodules_of, print_modules_classes_methods_attributes, get_doc

# -- Debug logging --------------------------------------------------------------------------------------------------
log_file_name = '../build/pandas_info.log'

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

exclude_attributes = [
]


# -- Implements custom skip functions for the parser ----------------------------------------------------------------
def _skip_pandas_module(mod, mod_name):
    return mod_name in exclude_modules or not mod_name.startswith('pandas')


def _skip_pandas_class(cls, cls_name):
    return cls_name in exclude_classes


def _skip_pandas_method(method_name):
    return method_name in exclude_methods


def _skip_pandas_attribute(attr_name):
    return attr_name in exclude_attributes


if __name__ == "__main__":
    # Initialize logging
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG)
    logging.debug('****************** STARTING THE LOG *************************')
    logging.debug(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

    # Execute parser for Pandas
    inspected_modules = set()
    modules = []
    get_submodules_of(pandas, inspected_modules, modules, _skip_pandas_module, _skip_pandas_class,
                      _skip_pandas_method, _skip_pandas_attribute)

    # You may uncomment this line in case you want to print out generated methods and attributes
#    print_modules_classes_methods_attributes(modules)
