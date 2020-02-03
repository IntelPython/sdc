# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2020, Intel Corporation All rights reserved.
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

UNDERLINE_CHARS = ['-', '`', ':', '~', '^', '_', '*', '+', '#', '<', '>']  # Characters that can underline title

SDC_USR_GUIDE_HEADING_STR = 'Intel Scalable Dataframe Compiler User Guide'

SDC_USER_GUIDE_PANDAS_STR = 'Pandas API:'

SDC_DEV_GUIDE_HEADING_STR = 'Intel Scalable Dataframe Compiler Developer Guide'


def get_indent(text):
    """
    Returns indentation for a given ``text``.

    :param text: String, can be multi-line. Only first non-empty line is used to determine the indentation
    :return: Indentation (the number of whitespace characters)
    """
    lines = text.split('\n')
    while len(lines) > 0 and lines[0] == '':
        lines.pop(0)

    if len(lines) == 0:
        return 0  # Text was empty, indentation for empty text is 0

    n_stripped = len(lines[0].lstrip())  # Length of the string after stripping whitespaces on the left
    return len(lines[0]) - n_stripped


def reindent(old_text, new_indent):
    """
    Perform re-indentation of the text ``old_text`` with new indent ``new_indent``.

    :param old_text: Multi-line string for which re-indentation is performed
    :param new_indent: New indent
    :return: New multi-line text
    """

    if old_text == '':
        return ' '*new_indent

    old_indent = get_indent(old_text)
    lines = old_text.split('\n')
    new_text = ''
    for line in lines:
        if line.strip() == '':
            new_text += '\n'
        else:
            line = line[old_indent:]
            new_text += ' '*new_indent + line + '\n'

    # If ``old_text`` has no ``'\n'`` in the end, remove it too from the ``new_text``
    if old_text[-1] != '\n':
        new_text = new_text[:-1]

    return new_text


def create_heading_str(title, underlying_symbol='-'):
    """
    Creates heading string for a given ``title``. Second line under title is decorated with ``underlying_symbol``

    Heading is created taking into account of ``title`` indentation.

    :param title:
    :param underlying_symbol:
    :return: resulting heading string
    """
    indent = get_indent(title)
    n = len(title.strip())
    return title + '\n' + ' '*indent + underlying_symbol*n


def get_docstring(obj):
    """
    Returns docstring for a given object or empty string if no-object is provided or there is no docstring for it.

    :param obj: Object for which the docstring to be provided
    :return: Docstring
    """
    if obj is None:
        return ''

    doc = obj.__doc__
    if doc is None:
        return ''
    else:
        return doc


def is_section_title(line, underline):
    """
    Checks whether line and consecutive underline form valid section title.

    .. note::
        Function expects leading and trailing whitespaces removed for both strings prior to the call.

    :param line: String, title text
    :param underline: String, underlying characters
    :return: True if line and underline form valid section title
    """

    if line is None:
        return False

    if underline is None:
        return False

    if line == '':
        return False

    if underline == '':
        return False

    n = len(line)
    for c in UNDERLINE_CHARS:
        s = c * n
        if underline == s:
            return True

    return False


def is_sdc_user_guide_header(sdc_header):
    """
    Checks whether a given title-text tuple forms valid Intel SDC header for User Guide.

    The header is expected to be 4 lines long, where the first three lines are of the form:
        Intel Scalable Dataframe Compiler User Guide
        ********************************************
        Pandas API: <pandas API name>
    The fourth line must be empty

    :param sdc_header: Tuple (title, text)
    :return: True if sdc_header forms valid Intel SDC User Guide docstring header
    """
    title, text = sdc_header
    return title.strip() == SDC_USR_GUIDE_HEADING_STR and text.strip().startswith(SDC_USER_GUIDE_PANDAS_STR)


def is_sdc_dev_guide_header(sdc_header):
    """
    Checks whether a given title-text tuple forms valid Intel SDC header for Developer Guide.

    The header is expected to be 3 lines long, where the first two lines are of the form:
        Intel Scalable Dataframe Compiler Developer Guide
        *************************************************
    The third line must be empty

    :param sdc_header: Tuple (title, text)
    :return: True if sdc_header forms valid Intel SDC Developer Guide docstring header
    """
    title, text = sdc_header
    return title.strip() == SDC_DEV_GUIDE_HEADING_STR


def extract_pandas_name_from(text):
    """
    Extracts Pandas API from ``text``.

    This function is used in conjunction with :func:`split_title`, which returns the tuple (title, text).
    The ``title`` must contain valid Intel SDC header. The ``text`` is expected to be in the form
    ``Pandas API: *fully qualified Pandas name*``

    :param text:
    :return: Pandas API name as a string
    """
    line = text.strip().split('\n', 1)[0]  # Pandas API is in the first line. Ignore whitespaces
    return line.replace(SDC_USER_GUIDE_PANDAS_STR, '').strip()  # Name begins right after ``Pandas API:``


def split_title(section):
    """
    Split section into title and remaining text.

    :param section: String, documented section
    :return: Tuple (title, text)
    """

    if section is None:
        return '', ''

    section = section.lstrip('\n')  # Remove leading empty lines

    lines = section.split('\n', 2)
    if len(lines) > 1:
        # Only sections with number of lines >= 2 can be a title
        if is_section_title(lines[0].strip(), lines[1].strip()):
            if len(lines) > 2:
                return lines[0], lines[2]  # First line is title, second is underline, remaining is text
            else:
                return lines[0], ''  # First line is title, second line is underline, but the text is empty string
        else:
            return '', section  # First two lines do not form valid heading
    else:
        return '', section  # When section is less than 3 lines we consider it having no title


def _merge_paragraphs_within_section(sections):
    """
    Internal utility function that merges paragraphs into a single section.

    This function call is required after initial splitting of the docstring into sections.  The initial split
    is based on the presence of ``'\n\n'``, which separates sections and paragraphs. The difference between
    section and paragraph is that section starts with the title of the form:

        This is title
        -------------
        This is the first paragraph. It may be multi-line.
        This is the second line of the paragraph.

        This is another multi-line paragraph.
        This is the second line of the paragraph.

    Special treatment is required for Intel SDC header section and the following description section. Intel SDC
    header section must the the first one in the docstring. It consists of exactly 3 lines:

        Intel Scalable Dataframe Compiler User Guide
        ********************************************
        Pandas API: *pandas_api_fully_qualified_name*

    Right after the Intel SDC header section the description section (if any) goes. It generally consists of two
    or more paragraphs. The first paragraph represents short description, which is typically single line.
    The following paragraphs provide full description. In rare cases documentation does not have description section,
    and this must be treated accordingly.


    :param sections: List of tuples ``(title, text)``.
    :return: Reformatted list of tuples ``(title, text)`, where paragraphs belonging to one section are merged in
        single ``text`` item.
    """
    if len(sections) == 0:
        return sections

    merged_sections = []
    # Check if the very first section is Intel SDC header
    section_title, section_text = sections[0]
    if is_sdc_user_guide_header((section_title, section_text)):
        merged_sections.append(sections[0])
        sections.pop(0)

    # Check if the next section is the short description
    section_title, section_text = sections[0]
    if section_title.strip() == '':
        merged_sections.append(sections[0])
        sections.pop(0)

    if len(sections) == 0:
        return merged_sections

    # Merge next sections with empty title into a single section representing full description
    section_title, section_text = sections[0]
    if section_title.strip() == '':
        sections.pop(0)
        while len(sections) > 0:
            title, text = sections[0]
            if title.strip() == '':
                section_text += '\n\n' + text
                sections.pop(0)
            else:
                break
        merged_sections.append((section_title, section_text))

    # Now merge paragraphs of remaining titled sections
    while len(sections) > 0:
        section_title, section_text = sections[0]
        sections.pop(0)
        while len(sections) > 0:
            title, text = sections[0]
            if title.strip() == '':
                section_text += '\n\n' + text
                sections.pop(0)
            else:
                break
        merged_sections.append((section_title, section_text))

    return merged_sections


def split_in_sections(doc):
    """
    Splits the doc string into sections

    Each section is separated by empty line. Sections can start with headers or without. Each header follows NumPy
    style:

        Section Title
        -------------

    Other permitted characters can be used to underline section title

    :param doc: Docstring to be split into sections
    :return: List, sections of the doc. Each section is a tuple of strings (title, text)

    :seealso: NumPy style `example
        <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy>`_
    """
    doc = reindent(doc, 0)
    sections = doc.split('\n\n')  # Sections are separated by empty lines
    titled_sections = []

    while len(sections) > 0:
        title, text = split_title(sections[0])
        sections.pop(0)
        titled_sections.append((title, text))

    return _merge_paragraphs_within_section(titled_sections)


def get_short_description(obj, sdc_header_flag=False):
    """
    Returns short description for a given object obj

    :param obj: Object for which short description needs to be returned
    :param sdc_header_flag: Flag indicating that the first three lines must be considered as Intel SDC header
    :return: String, short description
    :raises: NameError, when ``sdc_header_flag==True`` and no Intel SDC header section found.
        The header is expected to be 4 lines long, where the first three lines are of the form:
            Intel Scalable Dataframe Compiler User Guide
            ********************************************
            Pandas API: <pandas API name>
        The fourth line must be empty

    """
    doc = get_docstring(obj)
    if doc == '':
        return doc

    sections = split_in_sections(doc)  # tuple (title, text)

    if sdc_header_flag:
        if len(sections) > 1:  # There must be at least one more section after Intel SDC header section
            if not is_sdc_user_guide_header(sections[0]):
                raise NameError('No Intel SDC header section found')

            sections.pop(0)  # Ignore Intel SDC header section

    if len(sections) == 0:
        return ''  # Docstring has no sections, i.e. short description is absent

    title, text = sections[0]  # Short description is the first section of the docstring
    text = text.strip()
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    lines = ' '.join(lines)

    return lines


def cut_sdc_dev_guide(doc):
    """
    Removes Intel SDC Developer Guide related sections from the docstring.

    It is assumed that Developer Guide docstring follows the User Guide related sections of the docstring.
    Everything after section the titled *Intel Scalable Dataframe Compiler Developer Guide* is cut

    :param doc: Docstring that includes User Guide and the following Developer Guide sections
    :return: Docstring with the cut Developer Guide sections
    """
    sections = split_in_sections(doc)    # tuple (title, text)
    trimmed_sections = []

    while len(sections) > 0:
        if is_sdc_dev_guide_header(sections[0]):
            break
        trimmed_sections.append(sections[0])
        sections.pop(0)

    return trimmed_sections
