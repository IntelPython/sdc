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
from pathlib import Path

from sdc_object_utils import get_sdc_object_by_pandas_name, init_sdc_structure
from sdc_doc_utils import get_docstring, reindent, split_in_sections
from apiref_generator import (APIREF_TEMPLATE_FNAMES, reformat)


EXAMPLES_REL_PATH = Path('.') / '_examples'


def get_obj_examples(pandas_name):
    """
    Get list of examples for Pandas object.

    :param pandas_name: Pandas object for which documentation to be generated.
    :return: Generated docstring.
    """
    sdc_obj = get_sdc_object_by_pandas_name(pandas_name)
    sdc_doc = get_docstring(sdc_obj)
    sections = split_in_sections(reindent(sdc_doc, 0))
    sections_as_dict = {title.strip(): text for title, text in sections}
    example_section = sections_as_dict.get('Examples')
    if not example_section:
        return None

    examples = []
    section_names = ['literalinclude', 'command-output']
    for subsection in example_section.strip().split('\n\n'):
        subsection = subsection.strip()
        if any(subsection.startswith(f'.. {name}') for name in section_names):
            # remove a directory level from path to examples
            examples.append(subsection.replace(' ../', ' '))

    return reformat('\n\n'.join(examples))


def get_tmpl_examples(fname_templ):
    """Get all examples based on input template rst file"""
    tmpl_examples = []
    with open(fname_templ, encoding='utf-8') as fin:
        doc = fin.readlines()

        while len(doc) > 0:
            # Parsing lines until ``.. sdc_toctree`` section is met
            while len(doc) > 0 and not doc[0].startswith('.. sdc_toctree'):
                line = doc[0]
                if line.startswith('.. currentmodule::'):
                    current_module_name = line[19:].strip()
                doc.pop(0)

            if len(doc) == 0:
                break

            doc.pop(0)  # Skipping ``.. sdc_toctree``

            # Parsing the list of APIs
            while len(doc) > 0 and doc[0].strip() != '':
                line = doc[0]
                line = line.strip()
                full_name = current_module_name + '.' + line
                doc.pop(0)

                obj_examples = get_obj_examples(full_name)
                if obj_examples:
                    tmpl_examples.append(obj_examples)

            if len(doc) == 0:
                break

    return tmpl_examples


def generate_examples():
    """
    Master function for examples list generation.

    This function initializes SDC data structures, and parses required templates for
    Final RST file generation that lists all the examples.
    """
    init_sdc_structure()

    all_examples = []
    for templ_fname in APIREF_TEMPLATE_FNAMES:
        all_examples += get_tmpl_examples(templ_fname)

    if not EXAMPLES_REL_PATH.exists():
        EXAMPLES_REL_PATH.mkdir(parents=True, exist_ok=True)

    examples_rst_path = EXAMPLES_REL_PATH / 'examples.rst'
    with examples_rst_path.open('w', encoding='utf-8') as fd:
        for examples in all_examples:
            fd.write(examples + '\n')


if __name__ == "__main__":
    generate_examples()
