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
from sdc_object_utils import init_pandas_structure, init_sdc_structure, init_pandas_sdc_dict, get_sdc_object, get_obj
from sdc_object_utils import get_class_methods, get_class_attributes, get_fully_qualified_name
from sdc_doc_utils import is_sdc_user_guide_header, get_indent, reindent, get_short_description
from sdc_doc_utils import split_in_sections, get_docstring, create_heading_str, cut_sdc_dev_guide
import os

APIREF_REL_PATH = './_api_ref/'


def reformat(text):
    """
    Wrapper function that includes series of transformations of the ``text`` to fix Pandas docstrings which
    cause Sphinx to generate warnings.

    :param text: Original text with warnings
    :return: Modified text that fixes warnings
    """
    text = reformat_replace_star_list_with_dash_list(text)  # Must be called before :func:`reformat_asterisks`
    text = reformat_asterisks(text)  # Fix for * and ** symbols
    text = reformat_explicit_markup(text)  # Fix for explicit markup without a blank line
    text = reformat_bullet_list(text)  # Fix bullet list indentation issues
    text = reformat_remove_unresolved_references(text)  # Fix unresolved references after removal of References sections
    return reformat_remove_multiple_blank_lines(text)


def reformat_remove_unresolved_references(text):
    """
    Fixes unresolved references after removing References sections.

    Searches for pattern [numeric]_ in the text and removes it. Intel SDC references do not use [numeric]_ pattern

    :param text: Original text
    :return: Reformatted text
    """
    new_text = ''
    while len(text) > 0:
        idx = text.find('[')

        if idx >= 0:
            new_text += text[0:idx]
            idx1 = idx+1
            while idx1 < len(text) and text[idx1].isnumeric():
                # Iterating through numeric characters
                idx1 += 1

            if idx1+1 < len(text):
                # There are at least two more symbols after numeric ones in the text
                if text[idx1:idx1+2] != ']_':
                    new_text += text[idx:idx1+2]

                if idx1+2 < len(text):
                    text = text[idx1+2:]  # Remove reference
                else:
                    text = ''
            else:
                new_text += text[idx:]
                text = ''
        else:
            new_text += text
            text = ''
    return new_text


def reformat_replace_star_list_with_dash_list(text):
    """
    Replaces bullet lists starting with `*` with the lists starting with `-`

    :param text: Original text
    :return: New text without `*` bullet lists
    """
    lines = text.split('\n')
    new_text = ''
    for line in lines:
        if line.strip().startswith('* '):
            line = line.replace('* ', '- ', 1)

        new_text += line + '\n'

    return new_text


def reformat_remove_multiple_blank_lines(text):
    """
    Removes redundant blank lines

    After multiple passes of the text reformatting there could be redundant blank lines between sections.
    This pass is intended for removal of consecutive blank lines and keeping just one blank line between sections

    :param text: Original text
    :return: Text with removed redundant blank lines
    """

    len_changed = True

    while len_changed:
        new_text = text.replace('\n\n\n', '\n\n')
        len_changed = len(new_text) < len(text)
        text = new_text

    return new_text


def reformat_bullet_list(text):
    lines = text.split('\n')
    new_text = ''
    bullet_indent = -1
    while len(lines) > 0:
        line = lines[0]
        if line.strip().startswith('- '):
            # Here if met new bullet
            bullet_indent = get_indent(line)  # We need to know indent to identify multi-line bullets
            new_text += line + '\n'
        elif line.strip() == '':
            bullet_indent = -1  # We finished parsing multi-line bullet
            new_text += '\n'
        else:
            if bullet_indent >= 0:
                # Here if we're parsing multi-line bullet
                new_text += reindent(line, bullet_indent + 4) + '\n'
            else:
                # Here if we are not in bullet list
                new_text += line + '\n'
        lines.pop(0)

    return new_text


def reformat_explicit_markup(text):
    """
    Fixes Pandas docstring warning about explicit markup not followed by a blank line.

    Parses the text and finds ``'.. '`` strings by adding a blank line next after.

    :param text: Original text with warnings
    :return: Modified text that fixes warnings
    """
    lines = text.split('\n')
    new_text = ''
    while len(lines) > 0:
        line = lines[0]

        if line.strip().startswith('.. versionchanged') or line.strip().startswith('.. versionadded') or \
                line.strip().startswith('.. deprecated'):
            new_text += line + '\n'
            # Here if found explicit markup
            if len(lines) > 1:
                # Here if there is at least one line after explicit markup
                if lines[1].strip != '':
                    # Here if there is no empty line after explicit markup. Add new line then
                    new_text += '\n'
                lines.pop(0)
        elif line.strip().startswith('.. note') or line.strip().startswith('.. warning'):
            new_text += line.strip() + '\n'
            if len(lines) > 1:
                # Here if there is at least one line after explicit markup
                if lines[1].strip() == '':
                    # Here if there is empty line after explicit markup. Remove new line then
                    lines.pop(1)
        elif line.strip().startswith('.. ['):
            new_text += '\n'  # Remove references
        else:
            new_text += line + '\n'
        lines.pop(0)
    return new_text


def reformat_asterisks(text):
    """
    Fixes Pandas docstring warning about using * and ** without ending \* and \*\*.

    The fix distinguishes single * and ** by adding \\ to them. No changes for *italic* and **bold** usages.

    :param text: Original text with warnings
    :return: Modified text that fixes warnings
    """
    lines = text.split('\n')
    new_text = ''
    for line in lines:
        idx = 0  # Starting parsing position within the ``line``
        while idx < len(line):  # Parsing until end of string reached
            idx1 = line.find('*', idx)
            if idx1 >= idx:
                # There is at least one asterisk in the line
                idx2 = line.find('*', idx1+1)

                if idx2 == -1:
                    # Only one single asterisk in the line - Reformat to `\*`
                    line = line.replace('*', '\\*')
                    idx = len(line)  # Parsed the line. Go to another line
                elif idx2 == idx1+1:
                    # First double asterisk met in the line
                    idx2 = line.find('**', idx1+2)
                    if idx2 == -1:
                        # Only one double asterisk in the line Reformat to `\*\*`. But there could be more asterisks
                        line = line.replace('**', '\\*\\*')
                        idx = idx1+4
                    else:
                        # At least two double asterisks in the line
                        idx = idx2+2  # Deal with remaining asterisks on the next ``while`` loop iteration
                else:
                    # There is another asterisk apart from the first asterisk
                    if idx2+1 < len(line):
                        # There is at least one more symbol in the line after second asterisk
                        if line[idx2+1] == '*':
                            # Situation when double asterisk is met after the first single asterisk - Reformat to `\*`
                            line = line.replace('*', '\\*', 1)  # Replace the first single asterisk
                            idx = idx2  # Handle double asterisk on the next ``while`` iteration
                        else:
                            # Two asterisks met in the line to italize characters between them
                            idx = idx2+1
                    else:
                        # Second asterisk was the last symbol in the line
                        idx = len(line)
            else:
                # No asterisks in the line
                idx = len(line)
        new_text += line + '\n'

    return new_text


def reformat_pandas_params(title, text):
    """
    Re-formats ``text`` written in NumPy style documenting Parameters, Returns, Raises sections into
    explicit `:<param>:` style.

    Algorithm searches for the pattern:
    `<alpha_numeric_value> : <text>`
         `<text continued with indent>`
         `<text continued with indent>`
    Reformat to the following:
    `:<alpha_numeric_value>:`
         `<text>`
         `<text continued with indent>`
         `<text continued with indent>`


    :param title:
    :param text:
    :return: Reformatted text
    """

    # Internal function. Returns correct markup for :param <param>:, :return:, and :raises <exception>:
    def _get_param_text(title, param):
        title = title.strip()
        if title == 'Parameters':
            return ':param ' + param + ':'
        elif title == 'Return' or title == 'Returns':
            return ':return:'
        elif title == 'Raises':
            return ':raises:'

    # Internal function. Returns correct markup for Parameters section
    def _reformat_parameters(title, text):
        lines = text.split('\n')
        new_text = ''

        if len(lines) == 0:
            return new_text

        indent = get_indent(text)
        param = ''
        description = ''
        while len(lines) > 0:
            line = lines[0]
            line = line.strip()
            idx = line.find(' : ')
            if idx >= 0 & line[0:idx].isalnum():
                # Check if previous parameter existed. If so, need to add it to reformatted text
                if param != '':
                    new_text += _get_param_text(title, param) + '\n' + reindent(description, indent+4) + '\n'

                # Found parameter. Extract the description (can be multi-line)
                param = line[0:idx]
                description = line[idx+3:] + '\n'
                lines.pop(0)
            else:
                # There is no parameter description starting in this line.
                # Check if it is continuation of parameter description from previous lines
                if param != '':
                    # It is continuation of multi-line parameter description
                    description += reindent(line, indent+4) + '\n'
                else:
                    # This is not the description of parameter. Copy as is
                    new_text += reindent(line, indent) + '\n'
                lines.pop(0)

        if param != '' and description != '':
            new_text += _get_param_text(title, param) + '\n' + reindent(description, indent+4) + '\n'
        return new_text

    # Internal function. Returns correct markup for Raises section
    def _reformat_raises(title, text):
        lines = text.split('\n')
        new_text = ''

        if len(lines) == 0:
            return new_text

        indent = get_indent(text)
        param = ''
        description = ''
        while len(lines) > 0:
            line = lines[0]
            line = line.strip()

            # Check if it is continuation of parameter description from previous lines
            if param != '':
                # It is continuation of multi-line parameter description
                description += reindent(line, indent + 8) + '\n'
            else:
                # This is the first line of ``raises`` description
                param = _get_param_text(title, '') + '\n' + reindent(line, indent + 4)
                new_text += param + '\n'
            lines.pop(0)

        if param != '' and description != '':
            new_text += reindent(description, indent + 8) + '\n'
        return new_text + '\n'

    # Internal function. Returns correct markup for Returns section
    def _reformat_returns(title, text):
        lines = text.split('\n')
        new_text = ''

        if len(lines) == 0:
            return new_text

        indent = get_indent(text)
        param = ''
        description = ''
        while len(lines) > 0:
            line = lines[0]
            line = line.strip()

            # Check if it is continuation of parameter description from previous lines
            if param != '':
                # It is continuation of multi-line parameter description
                description += reindent(line, indent + 4) + '\n'
            else:
                # This is the first line of ``return`` description
                param = _get_param_text(title, '') + ' ' + line
                new_text += reindent(param, indent) + '\n'
            lines.pop(0)

        if param != '' and description != '':
            new_text += reindent(description, indent + 4) + '\n'
        return new_text + '\n'

    if title.strip() == 'Parameters':
        return _reformat_parameters(title, text)
    elif title.strip() == 'Returns' or title.strip() == 'Return':
        return _reformat_returns(title, text)
    elif title.strip() == 'Raises':
        return _reformat_raises(title, text)
    else:
        return text


def generate_simple_object_doc(pandas_obj, short_doc_flag=False, doc_from_pandas_flag=True, add_sdc_sections=True,
                               unsupported_warning=True, reformat_pandas=True):
    """
    Generates documentation for Pandas object obj according to flags.

    For complex objects such as modules and classes the function does not go to sub-objects,
    i.e. to class attributes and sub-modules of the module.

    :param pandas_obj: Pandas object for which documentation to be generated.
    :param short_doc_flag: Flag to indicate that only short description for the object is needed.
    :param doc_from_pandas_flag: Flag to indicate that the documentation must be taken from Pandas docstring.
           This docstring can be extended with Intel SDC specific sections. These are See Also, Examples,
           Notes, Warning, Limitations, etc. if ``add_sdc_sections`` flag is set.
    :param add_sdc_sections: Flag to indicate that extra sections of the documentation need to be taken from Intel SDC.
           If ``doc_from_pandas_flag==False`` then the description section is taken from Intel SDC too. Otherwise
           Intel SDC description section will be cut and Pandas API description will be used instead.
    :param unsupported_warning: Flag, if ``True`` includes warning message if corresponding Intel SDC object is not
           found. This indicates that given SDC method is unsupported.
    :param reformat_pandas: Flag, if ``True`` re-formats Parameters section to :param: style. Needed to work around
           Sphinx generator issues for Pandas Parameters section written in NumPy style
    :return: Generated docstring.
    """

    doc = ''
    if pandas_obj is None:
        return doc  # Empty documentation for no-object

    if doc_from_pandas_flag:  # Check if documentation needs to be generated from Pandas docstring
        if short_doc_flag:  # Check if only short description is needed
            doc = get_short_description(pandas_obj)  # Short description is requested
        else:
            # Exclude Examples, Notes, See Also, References sections
            sections = split_in_sections(reindent(get_docstring(pandas_obj), 0))
            while len(sections) > 0:
                title, text = sections[0]
                if title.strip() == '':  # Description sections
                    doc += text + '\n\n'
                    sections.pop(0)
                elif title.strip() == 'Examples':  # Exclude Examples section
                    sections.pop(0)
                elif title.strip() == 'Notes':  # Exclude Notes section (may be too specific to Pandas)
                    sections.pop(0)
                elif title.strip().lower() == 'see also':  # Exclude See Also section (may be too specific to Pandas)
                    sections.pop(0)
                elif title.strip() == 'References':  # Exclude References section (may be too specific to Pandas)
                    sections.pop(0)
                elif title.strip() == 'Parameters' or title.strip() == 'Raises' or title.strip() == 'Return' or \
                        title.strip() == 'Returns':
                    if reformat_pandas:
                        doc += reformat_pandas_params(title, text)
                        sections.pop(0)
                    else:
                        doc += create_heading_str(title) + '\n\n' + text + '\n\n'
                        sections.pop(0)
                else:
                    doc += create_heading_str(title) + '\n\n' + text + '\n\n'
                    sections.pop(0)

    if not add_sdc_sections:
        if reformat_pandas:
            return reformat(doc)
        else:
            return doc

    # Here if additional sections from Intel SDC object needs to be added to pandas_obj docstring
    sdc_obj = get_sdc_object(pandas_obj)
    if sdc_obj is None:
        if unsupported_warning:
            if reformat_pandas:
                doc = reformat(doc)

            if short_doc_flag:
                return doc + ' **Unsupported by Intel SDC**.'
            else:
                return doc + '\n\n.. warning::\n    This feature is currently unsupported ' \
                                      'by Intel Scalable Dataframe Compiler\n\n'

    if not short_doc_flag:
        sdc_doc = get_docstring(sdc_obj)
        sdc_doc = cut_sdc_dev_guide(sdc_doc)

        # Cut description section from ``sdc_doc``
        if is_sdc_user_guide_header(sdc_doc[0]):  # First section is SDC User Guide header
            sdc_doc.pop(0)

        if doc_from_pandas_flag:
            # Ignore description from Intel SDC, keep Pandas description only
            while len(sdc_doc) > 0:
                title, text = sdc_doc[0]
                if title.strip() != '':
                    break
                sdc_doc.pop(0)

        indent = get_indent(doc)
        for title, text in sdc_doc:
            if title.strip() == '':
                doc += '\n' + reindent(text, indent)
            else:
                doc += '\n' + reindent(create_heading_str(title), indent) + '\n' + \
                       reindent(text, indent) + '\n'

    return reformat(doc)


def get_rst_filename(obj_name):
    """
    Returns rst file name by respective object name.

    :param obj_name: String, object name for which file name is constructed
    :return: String, rst file name for the object being documented
    """
    file_name = obj_name.replace('.', '/')
    file_name = APIREF_REL_PATH + file_name + '.rst'
    return file_name


def open_file_for_write(file_name):
    """
    Opens file ``filename`` for writing. If necessary, creates file directories on the path.

    :param file_name: Absolute or relative path that includes file name being created.
    :return: File descriptor created.
    """
    directory = os.path.dirname(file_name)

    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory)

    return open(file_name, 'w', encoding='utf-8')


def write_rst(file_name, docstring):
    """
    Writes ``docstring`` into the file ``file_name``.

    :param file_name: String, name of the file including relative or absolute path
    :param docstring: String, docstring to be written in the file
    """
    file = open_file_for_write(file_name)
    file.write(docstring)
    file.close()


def write_simple_object_rst_file(pandas_name, short_doc_flag=False, doc_from_pandas_flag=True, add_sdc_sections=True):
    """
    Writes Pandas object ``pandas_name`` (e.g. 'pandas.Series.at') into rst file.

    RST file has the name derived from ``pandas_name`` (e.g. 'pandas.Series.at.rst'). Additional flags are used
    to control look and feel of the resulting content of the file. See :func:`generate_simple_object_doc` function
    for details about these flags.

    :param pandas_name: String, the name of Pandas object
    :param short_doc_flag: Flag, if ``True``, write short description of the object only
    :param doc_from_pandas_flag: Flag, if ``True``, derive the description from Pandas docstring for the object.
    :param add_sdc_sections: Flag, if ``True``, extend the docstring with respective Intel SDC sections (if any)
    """
    pandas_obj = get_obj(pandas_name)
    doc = generate_simple_object_doc(pandas_obj, short_doc_flag, doc_from_pandas_flag, add_sdc_sections)
    if doc is None or doc == '':
        return

    fname = get_rst_filename(pandas_name)
    write_rst(fname, doc)


def parse_templ_rst(fname_templ):
    """
    Parses input template rst file and outputs the final rst file
    Template document must have the following structure:

    Heading or subheading
    *********************

    Any text (if any)

    Another heading or subheading
    -----------------------------

    Any text (if any)

    .. currentmodule:: <module name>

    .. sdc_toctree
    <api1>
    <api2>
    <api3>
    ...

    Any text (if any)

    Any text (if any)

    Another heading or subheading
    -----------------------------

    Any text (if any)
    ...

    :param fname_templ:
    """
    path, fname_out = os.path.split(fname_templ)
    fname_out = fname_out.replace('_templ', '')
    fname_out = fname_out.replace('_', '', 1)
    fout = open_file_for_write(APIREF_REL_PATH + fname_out)
    with open(fname_templ, 'r', encoding='utf-8') as fin:
        doc = fin.readlines()

        while len(doc) > 0:
            # Parsing lines until ``.. sdc_toctree`` section is met
            while len(doc) > 0 and not doc[0].startswith('.. sdc_toctree'):
                line = doc[0]
                if line.startswith('.. currentmodule::'):
                    current_module_name = line[19:].strip()
                fout.write(line)
                doc.pop(0)

            if len(doc) == 0:
                return

            doc.pop(0)  # Skipping ``.. sdc_toctree``

            # Parsing the list of APIs
            while len(doc) > 0 and doc[0].strip() != '':
                line = doc[0]
                indent = get_indent(line)
                line = line.strip()
                full_name = current_module_name + '.' + line
                obj = get_obj(full_name)
                short_description = generate_simple_object_doc(obj, short_doc_flag=True).strip()
                new_line = reindent(':ref:`', indent) + line + ' <' + full_name + '>`\n' + \
                    reindent(short_description, indent+4) + '\n'
                fout.write(new_line)
                doc.pop(0)

                full_description = generate_simple_object_doc(obj, short_doc_flag=False)
                f = open_file_for_write(APIREF_REL_PATH + full_name + '.rst')
                f.write('.. _' + full_name + ':\n\n:orphan:\n\n')
                f.write(create_heading_str(full_name, '*') + '\n\n')
                f.write(full_description)
                f.close()

            if len(doc) == 0:
                return

    fout.close()


def write_class_rst_files(cls, short_doc_flag=False, doc_from_pandas_flag=True, add_sdc_sections=True):
    # Currenlty not in use. Should be used for auto-documenting class methods and attributes.

    for method_name, method_object in get_class_methods(cls):
        write_simple_object_rst_file(get_fully_qualified_name(cls) + '.' + method_name,
                                     short_doc_flag, doc_from_pandas_flag, add_sdc_sections)

    for attr_name, attr_object in get_class_attributes(cls):
        write_simple_object_rst_file(get_fully_qualified_name(cls) + '.' + attr_name,
                                     short_doc_flag, doc_from_pandas_flag, add_sdc_sections)


def generate_api_reference():
    init_pandas_structure()
    init_sdc_structure()
    init_pandas_sdc_dict()

    parse_templ_rst('./_templates/_api_ref.pandas.series_templ.rst')


if __name__ == "__main__":
    generate_api_reference()
