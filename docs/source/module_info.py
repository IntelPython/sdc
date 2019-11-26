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
from info import trim
import logging


# -- Debug logging --------------------------------------------------------------------------------------------------
# Logging information about attribute parsing
def _attribute_logging(s):
    logging.debug('[ATTRIBUTE]' + s)
    return


# Logging information about method parsing
def _method_logging(s):
    logging.debug('[METHOD]' + s)
    return


# Logging information about class parsing
def _class_logging(s):
    logging.debug('[CLASS]' + s)
    return


# Logging information about module parsing
def _module_logging(s):
    logging.debug('[MODULE]' + s)
    return


# -- Returns all classes and respective methods of the module -------------------------------------------------------
def get_submodules_of(module, inspected, module_list, skip_module_test, skip_class_test,
                      skip_method_test, skip_attribute_test):

    # Returns True if the mod module will not be included in API Reference
    def _skip_module(mod):
        mod_name = mod.__name__  # Get new submodule name
        sk_mod = False

        if mod in inspected:  # Ignore already traversed modules
            sk_mod = True
            _module_logging('`' + mod_name + '` already traversed. Ignoring')
            return sk_mod

        if '._' in mod_name or mod_name.startswith('_'):  # Ignore internal module
            sk_mod = True
            _module_logging('`' + mod_name + '` is internal (starts with _). Ignoring')
            return sk_mod

        if skip_module_test(mod, mod_name):
            sk_mod = True
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

        if skip_class_test(cls, class_name):
            sk_class = True
            return sk_class

        return sk_class

    # Returns True if the method method_name will not be included in API Reference
    def _skip_method(method_name):
        sk_method = False

        if method_name.startswith('_'):  # Ignore internal methods
            sk_method = True
            _method_logging('`' + method_name + '` is internal (starts with __). Ignoring')
            return sk_method

        if skip_method_test(method_name):
            sk_method = True
            return sk_method

        return sk_method

    # Returns True if the attribute attr_name will not be included in API Reference
    def _skip_attribute(attr_name):
        sk_attr = False

        if attr_name.startswith('_'):  # Ignore internal methods
            sk_attr = True
            _attribute_logging('`' + attr_name + '` is internal (starts with __). Ignoring')
            return sk_attr

        if skip_attribute_test(attr_name):
            sk_attr = True
            return sk_attr

        return sk_attr

    # Creates the list of methods for the class
    def _generate_class_methods(cls):
        meths = [func for func in dir(cls) if callable(getattr(cls, func)) and not _skip_method(func)]
        for meth in meths:
            _method_logging('Adding method `' + meth + '` to the list')
        return meths

    # Creates the list of class's attributes
    def _generate_class_attributes(cls):
        attrs = [func for func in dir(cls) if not callable(getattr(cls, func)) and not _skip_attribute(func)]
        for att in attrs:
            _attribute_logging('Adding attribute `' + att + '` to the list')
        return attrs

    # -- get_classes_of() implementation begins
    if _skip_module(module):
        return

    inspected.add(module)  # Add module to the set of traversed modules
    module_name = module.__name__
    module_list.append({'module_name': module_name, 'module_object': module, 'classes': []})

    _module_logging('********************** Inspecting module `' + module_name + '`')

    class_list = []
    # Traverses the mod module classes and submodules
    for (name, obj) in getmembers(module):  # Iterate through members of the submodule
        if isclass(obj):  # We are interested in members, which are classes
            if not _skip_class(obj):
                _class_logging('********************** Inspecting class `' + name + '`')
                methods = _generate_class_methods(obj)  # Inspect methods of the class of interest only
                attributes = _generate_class_attributes(obj)  # Inspect attributes of the class of interest only
                class_list.append({'class_name': name, 'class_object': obj, 'class_methods': methods,
                                   'class_attributes': attributes})
                module_list[-1]['classes'] = class_list

        if ismodule(obj):
            if not _skip_module(obj):
                get_submodules_of(obj, inspected, module_list, skip_module_test, skip_class_test,
                                  skip_method_test, skip_attribute_test)

    return


# -- Returns all classes and respective methods of the module -------------------------------------------------------
def print_modules_classes_methods_attributes(modules):
    for the_module in modules:  # modules is the list, each element represents dictionary characterizing the sub-module
        print(the_module['module_name'])
        for the_class in the_module['classes']:
            print('- ' + the_class['class_name'])
            print('  METHODS:')
            for the_method in the_class['class_methods']:
                print('    ' + the_method)
            print('  ATTRIBUTES:')
            for the_attribute in the_class['class_attributes']:
                print('    ' + the_attribute)
    return


# -- These symbols can be used to underline section title -----------------------------------------------------------
UNDERLINE_SYMBOLS = ['~', '#', '@', '^', '*', '-', '_', '+', '=']


# -- Split section into section title and remaining text ------------------------------------------------------------
def split_title(section):
    def _is_section_title(title_line, underscore_line):
        n = len(title_line)
        for c in UNDERLINE_SYMBOLS:
            s = c * n
            if underscore_line.startswith(s):
                return True

        return False

    trimmed = trim(section)
    lines = trimmed.split('\n', 2)
    if len(lines) > 2:
        # Only sections with number of lines>2 can start with a title
        if _is_section_title(lines[0], lines[1]):
            return lines[0], lines[2]
        else:
            return '', section
    else:
        return '', section


# -- Parse docstring by forming the list of sections, where each section is dictionary with title and text ----------
def parse_docstring(doc):
    sections = doc.split('\n\n')
    titled_sections = []

    # The first and the second sections are to be Short and Long description
    section = sections[0]
    title, text = split_title(section)
    titled_sections.append({'title': title, 'text': text})

    section = sections[1]
    title, text = split_title(section)
    titled_sections.append({'title': title, 'text': text})

    # Other sections. Merge those which are just separated by blank lines
    for i in range(2, len(sections)):
        section = sections[i]
        title, text = split_title(section)
        if title == '':
            titled_sections[-1]['text'] += '\n\n' + text
        else:
            titled_sections.append({'title': title, 'text': text})

    return titled_sections


# -- Get full documentation for the class cls -----------------------------------------------------------------------
def get_doc(cls):
    obj = cls['class_object']
    doc = parse_docstring(obj.__doc__)
    short_description_doc = doc[0]['text']
    long_description_doc = doc[1]['text']
    parameters_doc = ''.join([sec['text'] for sec in doc if sec['title'] == 'Parameters'])
    returns_doc = ''.join([sec['text'] for sec in doc if sec['title'] == 'Returns'])
    raises_doc = ''.join([sec['text'] for sec in doc if sec['title'] == 'Raises'])
    seealso_doc = ''.join([sec['text'] for sec in doc if sec['title'] == 'See also'])
    notes_doc = ''.join([sec['text'] for sec in doc if sec['title'] == 'Notes'])
    examples_doc = ''.join([sec['text'] for sec in doc if sec['title'] == 'Examples'])

    return {
        "name": cls['class_name'],
        "module": obj.__module__,
        "title": short_description_doc,
        "description": long_description_doc,
        "parameters": parameters_doc,
        "returns": returns_doc,
        "raises": raises_doc,
        "seealso": seealso_doc,
        "notes": notes_doc,
        "examples": examples_doc
    }
