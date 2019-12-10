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
import logging
import sys


# -- Debug logging --------------------------------------------------------------------------------------------------
ENABLE_LOGGING = False


# Logging information about attribute parsing
def _attribute_logging(s):
    if ENABLE_LOGGING:
        logging.debug('[ATTRIBUTE]' + s)
    return


# Logging information about method parsing
def _method_logging(s):
    if ENABLE_LOGGING:
        logging.debug('[METHOD]' + s)
    return


# Logging information about function parsing
def _function_logging(s):
    if ENABLE_LOGGING:
        logging.debug('[FUNCTION]' + s)
    return


# Logging information about class parsing
def _class_logging(s):
    if ENABLE_LOGGING:
        logging.debug('[CLASS]' + s)
    return


# Logging information about module parsing
def _module_logging(s):
    if ENABLE_LOGGING:
        logging.debug('[MODULE]' + s)
    return


# -- Returns all classes and respective methods of the module -------------------------------------------------------
def get_submodules_of(module, inspected, module_list, skip_module_test, skip_class_test,
                      skip_method_test, skip_attribute_test, skip_function_test):

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
    def _skip_method(cls, method_name):
        sk_method = False

        if method_name.startswith('_'):  # Ignore internal methods
            sk_method = True
            _method_logging('`' + method_name + '` is internal (starts with __). Ignoring')
            return sk_method

        if skip_method_test(cls, method_name):
            sk_method = True
            return sk_method

        return sk_method

    # Returns True if the method method_name will not be included in API Reference
    def _skip_function(function, function_name):
        sk_function = False

        if function_name.startswith('_'):  # Ignore internal function
            sk_function = True
            _function_logging('`' + function_name + '` is internal (starts with __). Ignoring')
            return sk_function

        if skip_function_test(function, function_name):
            sk_function = True
            return sk_function

        return sk_function

    # Returns True if the attribute attr_name will not be included in API Reference
    def _skip_attribute(cls, attr_name):
        sk_attr = False

        if attr_name.startswith('_'):  # Ignore internal methods
            sk_attr = True
            _attribute_logging('`' + attr_name + '` is internal (starts with __). Ignoring')
            return sk_attr

        if skip_attribute_test(cls, attr_name):
            sk_attr = True
            return sk_attr

        return sk_attr

    # Creates the list of methods for the class
    def _generate_class_methods(cls):
        meths = [func for func in dir(cls) if callable(getattr(cls, func)) and not _skip_method(cls, func)]
        for meth in meths:
            _method_logging('Adding method `' + meth + '` to the list')
        return meths

    # Creates the list of class's attributes
    def _generate_class_attributes(cls):
        attrs = [func for func in dir(cls) if not callable(getattr(cls, func)) and not _skip_attribute(cls, func)]
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
    module_list[-1]['classes'] = class_list
    function_list = []
    module_list[-1]['functions'] = function_list

    # Traverses the mod module classes and submodules
    for (name, obj) in getmembers(module):  # Iterate through members of the submodule
        if isclass(obj):  # We are interested in objects, which are classes
            if not _skip_class(obj):
                _class_logging('********************** Inspecting class `' + name + '`')
                methods = _generate_class_methods(obj)  # Inspect methods of the class of interest only
                attributes = _generate_class_attributes(obj)  # Inspect attributes of the class of interest only
                class_list.append({'class_name': name, 'class_object': obj, 'class_methods': methods,
                                   'class_attributes': attributes})

        if isfunction(obj):  # We are interested in objects, which are functions
            if not _skip_function(obj, name):
                function_list.append({'function_name': name, 'function_object': obj})

        if ismodule(obj):
            if not _skip_module(obj):
                get_submodules_of(obj, inspected, module_list, skip_module_test, skip_class_test,
                                  skip_method_test, skip_attribute_test, skip_function_test)

    return


# -- Returns all classes and respective methods of the module -------------------------------------------------------
def print_modules_classes_methods_attributes(modules):
    for the_module in modules:  # modules is the list, each element represents dictionary characterizing the sub-module
        print(the_module['module_name'])
        print('  FUNCTIONS:')
        for the_function in the_module['functions']:
            print('  - ' + the_function['function_name'])

        print('  CLASSES:')
        for the_class in the_module['classes']:
            print('  - ' + the_class['class_name'])
            print('    METHODS:')
            for the_method in the_class['class_methods']:
                print('      ' + the_method)
            print('    ATTRIBUTES:')
            for the_attribute in the_class['class_attributes']:
                print('      ' + the_attribute)
    return


# -- Trimming docstring  --------------------------------------------------------------------------------------------
def trim(docstring):
    # Copyright 2015: Mirantis Inc.
    # All Rights Reserved.
    #
    #    Licensed under the Apache License, Version 2.0 (the "License"); you may
    #    not use this file except in compliance with the License. You may obtain
    #    a copy of the License at
    #
    #         http://www.apache.org/licenses/LICENSE-2.0
    #
    #    Unless required by applicable law or agreed to in writing, software
    #    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
    #    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
    #    License for the specific language governing permissions and limitations
    #    under the License.

    """trim function from PEP-257"""
    if not docstring:
        return ""
    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = docstring.expandtabs().splitlines()
    # Determine minimum indentation (first line doesn't count):
    indent = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < sys.maxsize:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)

    # Current code/unittests expects a line return at
    # end of multiline docstrings
    # workaround expected behavior from unittests
    if "\n" in docstring:
        trimmed.append("")

    # Return a single string:
    return "\n".join(trimmed)


# -- String formatting ----------------------------------------------------------------------------------------------
def reindent(string):
    # Copyright 2015: Mirantis Inc.
    # All Rights Reserved.
    #
    #    Licensed under the Apache License, Version 2.0 (the "License"); you may
    #    not use this file except in compliance with the License. You may obtain
    #    a copy of the License at
    #
    #         http://www.apache.org/licenses/LICENSE-2.0
    #
    #    Unless required by applicable law or agreed to in writing, software
    #    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
    #    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
    #    License for the specific language governing permissions and limitations
    #    under the License.

    return "\n".join(l.strip() for l in string.strip().split("\n"))


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

    if section.startswith('\n'):
        section = section.replace('\n', '', 1)

    lines = section.split('\n', 2)
    if len(lines) > 2:
        # Only sections with number of lines>2 can start with a title
        if _is_section_title(lines[0].strip(), lines[1].strip()):
            return lines[0], lines[2]
        else:
            return '', section
    else:
        return '', section


# -- Parse docstring by forming the list of sections, where each section is dictionary with title and text ----------
def split_in_sections(doc, sdc_header_section_flag=False):
    sections = doc.split('\n\n')
    titled_sections = []

    # For SDC API Reference documentation the topmost section gives Pandas API name
    if sdc_header_section_flag:
        section = sections[0]
        title, text = split_title(section)
        titled_sections.append({'title': title, 'text': text})
        sections.pop(0)

    # Special processing for short and long description sections, if any
    section = sections[0]
    title, text = split_title(section)
    while title.strip() == '':
        titled_sections.append({'title': title, 'text': text})
        sections.pop(0)
        if len(sections) > 0:
            section = sections[0]
            title, text = split_title(section)
        else:
            break

    # Other sections. Merge those which are just separated by blank lines
    for i in range(len(sections)):
        section = sections[i]
        title, text = split_title(section)
        if title.strip() == '':
            titled_sections[-1]['text'] += '\n\n' + text
        else:
            titled_sections.append({'title': title, 'text': text})

    return titled_sections


def get_function_doc(func, sdc_header_flag=False):
    doc = func.__doc__

    if doc is None:
        doc = ''

    titled_sections = split_in_sections(doc, sdc_header_flag)
    return titled_sections


def get_function_short_description(func, sdc_header_flag=False):
    titled_sections = get_function_doc(func, sdc_header_flag)
    if sdc_header_flag: # Ignore the first section
        titled_sections.pop(0)
    short_description = titled_sections[0]['text']

    # Make it single line in case it is multi-line
    lines = short_description.split('\n')
    lines = [s.strip()+' ' for s in lines]
    short_description = ''.join(lines)

    return short_description


def create_header_str(s, underlying_symbol='*'):
    n = len(s)
    return s + '\n' + underlying_symbol*n


def get_function(func_name, modules):
    """
    Searches for the function func_name in the modules list. Name can or cannot be given fully qualified

    :param func_name: string, the function name being searched
    :param modules: the list of modules created by :func:`get_submodules_of`
    :return: function object or None
    """

    # Check if fully qualified name given
    if func_name.find('.') != -1:
        split_name = func_name.rsplit('.', 1)
        func_name = split_name[-1]
        module_name = split_name[-2]

        the_module = next((e for e in modules if e['module_name'] == module_name), None)
        try:
            if the_module:
                return getattr(the_module['module_object'], func_name)
            else:
                return None
        except AttributeError:
            return None
    else:
        for the_module in modules:
            for func_dict in the_module['functions']:
                if func_name == func_dict['function_name']:
                    return func_dict['function_object']

    return None

def get_method_attr(name, modules):
    """
    Searches for the method/attribute name in the modules list. Name is fully qualified

    :param name: string, the method/attribute being searched
    :param modules: the list of modules created by :func:`get_submodules_of`
    :return: method/attribute object or None
    """
    split_name = name.rsplit('.', 2)
    name = split_name[-1]
    class_name = split_name[-2]
    module_name = split_name[-3]

    the_module = next((e for e in modules if e['module_name'] == module_name), None)
    the_class = next((e for e in the_module['classes'] if e['class_name'] == class_name), None)
    try:
        return getattr(the_class['class_object'], name)
    except AttributeError:
        return None
