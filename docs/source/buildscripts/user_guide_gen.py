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

from module_info import get_function, get_method_attr, get_function_doc, get_function_short_description
from module_info import create_header_str
from pandas_info import get_pandas_modules, init_pandas_logging
from sdc_info import get_sdc_modules, init_sdc_logging
from texttable import Texttable
import os

PANDAS_API_STR = 'Pandas API: '  # This substring prepends Pandas API name in the documentation
APIREF_RELPATH = r'./_api_ref/'  # Relative path to API Reference folder
RST_MODULES = {
    'api_reference.rst': ['pandas'],
    'io.rst': ['pandas.io.api', 'pandas.io.clipboards', 'pandas.io.common', 'pandas.io.excel',
               'pandas.io.feather_format', 'pandas.io.formats.console', 'pandas.io.formats.format',
               'pandas.io.formats.printing', 'pandas.io.gbq', 'pandas.io.html', 'pandas.io.json',
               'pandas.io.msgpack', 'pandas.io.msgpack.exceptions', 'pandas.io.packers', 'pandas.io.parquet',
               'pandas.io.parsers', 'pandas.io.pickle', 'pandas.io.pytables', 'pandas.io.sas',
               'pandas.io.sas.sasreader', 'pandas.io.spss', 'pandas.io.sql', 'pandas.io.stata'],
    'series.rst': ['pandas.Series'],
    'dataframe.rst': ['pandas.DataFrame'],
    ''
    'general_functions.rst': [],
}

pandas_modules = []  # List of Pandas submodules along with its functions and classes
sdc_modules = []  # List of Intel SDC submodules along with its functions and classes


def generate_module_doc(the_module):
    module_doc = None
    module_name = the_module['module_name']

    # First, look up if there is RST file documenting particular module
    for rst in RST_MODULES:
        for mod in RST_MODULES[rst]:
            if mod == module_name:
                return module_doc  # If there is a documentation for a given module then just return

    # If there is no RST file then we create the documentation based on module's docstring
    module_obj = the_module['module_object']
    module_description = get_function_short_description(module_obj).strip()
    if module_description is None:
        module_description = ''

    module_doc = module_description + '\n\nFor details please refer to Pandas API Reference for :py:mod:`' + \
        module_name + '`\n\n'
    return module_doc


def generate_api_index_for_module(the_module):
    module_description = generate_module_doc(the_module)
    if module_description is None:
        module_description = ''
    module_doc = ''

    module_header_flag = False
    # Document functions first, if any
    tab = Texttable()
    for func in the_module['functions']:  # Iterate through the module functions
        name = func['function_name']
        obj = getattr(the_module['module_object'], name)  # Retrieve the function object
        description = get_function_short_description(obj).strip()
        tab.add_rows([[name, description]], header=False)

    module_name = ''
    func_doc = tab.draw()
    if func_doc and func_doc != '':  # If the function list is not empty then add module name to the document
        module_name = the_module['module_name']
        module_doc += create_header_str(module_name, '~') + '\n\n' + module_description + '\n\n' + \
            create_header_str('Functions:', '-') + \
            '\n\n' + func_doc + '\n\n'
        module_header_flag = True

    # Document classes
    classes_header_flag = False
    for the_class in the_module['classes']:  # Iterate through the module classes
        tab.reset()
        class_name = the_class['class_name']
        class_obj = the_class['class_object']
        class_description = class_obj.__doc__
        if not class_description:
            class_description = ''
        class_doc = ''
        class_header_flag = False

        # Document class attributes first, if any
        for attr in the_class['class_attributes']:  # Iterate through the class attributes
            name = attr
            obj = getattr(the_class['class_object'], name)  # Retrieve the attribute object
            description = get_function_short_description(obj).strip()
            tab.add_rows([[name, description]], header=False)

        attr_doc = tab.draw()
        if attr_doc and attr_doc != '':  # If the attribute list is not empty then add class name to the document
            class_header_flag = True
            class_doc += create_header_str(class_name, '^') + '\n\n' + class_description + '\n\n' + \
                create_header_str('Attributes:', '+') + \
                '\n\n' + attr_doc + '\n\n'

        # Document class methods, if any
        for method in the_class['class_methods']:  # Iterate through the class methods
            name = method
            obj = getattr(the_class['class_object'], name)  # Retrieve the method object
            description = get_function_short_description(obj).strip()
            tab.add_rows([[name, description]], header=False)

        method_doc = tab.draw()
        if method_doc and method_doc != '':  # If the method list is not empty then add class name to the document
            if not class_header_flag:
                class_doc += create_header_str(class_name, '^') + '\n\n' + class_description + '\n\n' + \
                             create_header_str('Methods:', '+') + \
                             '\n\n' + method_doc + '\n\n'
                class_header_flag = True
            else:
                class_doc += create_header_str('Methods:', '+') + \
                             '\n\n' + method_doc + '\n\n'

        if not module_header_flag:  # There is no module header yet
            if class_header_flag:  # There were methods/attributes for the class
                module_doc += create_header_str(module_name, '~') + '\n\n' + module_description + '\n\n' + \
                              create_header_str('Classes:', '-') + \
                              '\n\n' + class_doc + '\n\n'
                module_header_flag = True
                classes_header_flag = True
        else:  # The module header has been added
            if class_header_flag:  # There are new methods/attributes for the class
                if not classes_header_flag:  # First class of the module description
                    module_doc += create_header_str('Classes:', '-') + '\n\n'
                module_doc += '\n\n' + class_doc + '\n\n'
    return module_doc


def get_module_rst_fname(the_module):
    file_name = the_module['module_name']
    file_name = file_name.replace('.', '/')
    file_name = APIREF_RELPATH + file_name + '.rst'
    return file_name


def generate_api_index():
    doc = '.. _apireference::\n\nAPI Reference\n*************\n\n' \
          '.. toctree::\n   :maxdepth: 1\n\n'

    for the_module in pandas_modules:  # Iterate through pandas_modules
        module_doc = generate_api_index_for_module(the_module)
        if len(module_doc) > 0:
            file_name = get_module_rst_fname(the_module)
            write_rst(file_name, module_doc)
            doc += '   ' + file_name + '\n'
    return doc


def generate_sdc_object_doc(sdc_func):
    sdc_titled_sections = get_function_doc(sdc_func, True)
    sdc_see_also_text = next((sec['text'] for sec in sdc_titled_sections
                              if sec['title'].lower().strip() == 'see also'), '')
    sdc_limitations_text = next((sec['text'] for sec in sdc_titled_sections
                                 if sec['title'].lower().strip() == 'limitations'), '')
    sdc_examples_text = next((sec['text'] for sec in sdc_titled_sections
                              if sec['title'].lower().strip() == 'examples'), '')

    # Get respective Pandas API name
    pandas_name = sdc_titled_sections[0]['text'].strip()
    pandas_name = pandas_name.replace(PANDAS_API_STR, '')
    pandas_name = pandas_name.replace('\n', '')

    # Find respective Pandas API
    doc_object = get_method_attr(pandas_name, pandas_modules)
    if not doc_object:
        doc_object = get_function(pandas_name, pandas_modules)
    if not doc_object:
        raise NameError('Pandas API:' + pandas_name + 'does not exist')

    # Extract Pandas API docstring as the list of sections
    pandas_titled_sections = []
    if doc_object:
        pandas_titled_sections = get_function_doc(doc_object, False)

    # Form final docstring which is a combination of Pandas docstring for the description, Parameters section,
    # Raises section, Returns section. See Also, Limitations and Examples sections (if any) are taken from SDC docstring
    short_description_section = pandas_titled_sections[0]['text'] + '\n\n'
    pandas_titled_sections.pop(0)

    long_description_section = ''
    while pandas_titled_sections[0]['title'] == '':
        long_description_section += pandas_titled_sections[0]['text'] + '\n\n'
        pandas_titled_sections.pop(0)

    raises_section = parameters_section = returns_section = see_also_section = \
        limitations_section = examples_section = ''
    for section in pandas_titled_sections:
        title = section['title'].lower().strip()
        if title == 'raises':
            raises_section = 'Raises\n------\n\n' + section['text'] + '\n\n'
        elif title == 'parameters':
            parameters_section = 'Parameters\n----------\n\n' + section['text'] + '\n\n'
        elif title == 'return' or title == 'returns':
            returns_section = 'Returns\n-------\n\n' + section['text'] + '\n\n'

    if sdc_see_also_text:
        see_also_section = '\n.. seealso::\n\n' + sdc_see_also_text + '\n\n'

    if sdc_limitations_text:
        limitations_section = 'Limitations\n-----------\n\n' + sdc_limitations_text + '\n\n'

    if sdc_examples_text:
        examples_section = 'Examples\n-----------\n\n' + sdc_examples_text + '\n\n'

    rst_label = pandas_name.replace('.', '_')

    n = len(pandas_name)
    docstring = \
        '.. _' + rst_label + ':\n\n' + \
        pandas_name + '\n' + '*'*n + '\n' + \
        short_description_section + \
        long_description_section + \
        parameters_section + \
        returns_section + \
        raises_section +  \
        limitations_section +  \
        examples_section +  \
        see_also_section

    file_name = rst_label + '.rst'

    return file_name, docstring


def write_rst(file_name, docstring):
    directory = os.path.dirname(file_name)

    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory)

    file = open(file_name, 'w')
    file.write(docstring)
    file.close()


if __name__ == "__main__":
    init_pandas_logging()
    pandas_modules = get_pandas_modules()

    init_sdc_logging()
    sdc_modules = get_sdc_modules()

    for the_module in sdc_modules:
        if the_module['module_name'] == 'sdc.datatypes.hpat_pandas_series_functions':
            for func in the_module['functions']:
                file_name, doc = generate_sdc_object_doc(func['function_object'])
                write_rst(APIREF_RELPATH + file_name, doc)

    doc = generate_api_index()
    write_rst('apireference.rst', doc)
