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

import logging
import sdc
from datetime import datetime
from module_info import get_submodules_of, print_modules_classes_methods_attributes, ENABLE_LOGGING, trim
from module_info import get_function, get_function_doc

# -- String for pattern matching that indicates that the docstring belongs to Intel SDC API Reference ---------------
SDC_USR_GUIDE_HEADING_STR = \
    'Intel Scalable Dataframe Compiler User Guide********************************************'
SDC_DEV_GUIDE_HEADING_STR = \
    'Intel Scalablle Dataframe Compiler Developer Guide**************************************************'

# -- Debug logging --------------------------------------------------------------------------------------------------
log_file_name = '../build/sdc_info.log'


# -- Submodules, classes, and methods to be excluded from API Reference ---------------------------------------------
exclude_modules = [
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

exclude_classes = [
]

exclude_methods = [
]

exclude_attributes = [
]

exclude_functions = [
]


# -- Implements custom skip functions for the parser ----------------------------------------------------------------
def _skip_sdc_module(mod, mod_name):
    return mod_name in exclude_modules or (not mod_name.startswith('sdc') and not mod_name.startswith('hpat'))


def _skip_sdc_class(cls, cls_name):
    return True  # Exclude all classes
#    return cls_name in exclude_classes  # Explicit exclusion of the class


def _skip_sdc_method(cls, method_name):
    # Exclude the method if in the exclude_methods list
    if method_name in exclude_methods:
        return True

    # Exclude the method without docstring
    try:
        doc = getattr(cls, method_name).__doc__
        if len(doc) < 1:
            return True
    except AttributeError:
        return True
    except TypeError:
        return True

    # Exclude the method that does have docstring aimed for API Reference
    return not doc.startswith(SDC_USR_GUIDE_HEADING_STR)


def _skip_sdc_function(func, function_name):
    # Exclude the function if in the exclude_methods list
    if function_name in exclude_functions:
        return True

    # Exclude the function without docstring
    try:
        doc = func.__doc__
        if len(doc) < 1:
            return True
    except AttributeError:
        return True
    except TypeError:
        return True

    # Include the function that has docstring aimed for API Reference
    doc = ''.join(trim(doc).splitlines())
    return not doc.startswith(SDC_USR_GUIDE_HEADING_STR)


def _skip_sdc_attribute(cls, attr_name):
    # Exclude the attribute if in the exclude_methods list
    if attr_name in exclude_attributes:
        return True

    #  Exclude the attribute without docstring
    try:
        doc = getattr(cls, attr_name).__doc__
        if len(doc) < 1:
            return True
    except AttributeError:
        return True
    except TypeError:
        return True

    # Include the attribute that has docstring aimed for API Reference
    doc = ''.join(trim(doc).splitlines())
    return not doc.startswith(SDC_USR_GUIDE_HEADING_STR)


def get_sdc_modules():
    inspected_modules = set()
    modules = []
    get_submodules_of(sdc, inspected_modules, modules, _skip_sdc_module, _skip_sdc_class,
                      _skip_sdc_method, _skip_sdc_attribute, _skip_sdc_function)
    return modules


def init_sdc_logging():
    if ENABLE_LOGGING:
        logging.basicConfig(filename=log_file_name, level=logging.DEBUG)
        logging.debug('****************** STARTING THE LOG *************************')
        logging.debug(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))


if __name__ == "__main__":
    # Initialize logging
    init_sdc_logging()

    # Execute parser for SDC

    # You may uncomment this line in case you want to print out generated methods and attributes
    # print_modules_classes_methods_attributes(modules)
    modules = get_sdc_modules()

    func = get_function('hpat_pandas_series_at', modules)
    if func:
        titled_sections = get_function_doc(func)
        print(titled_sections)
