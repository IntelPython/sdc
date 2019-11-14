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


# coding: utf-8
# Configuration file for the Sphinx documentation builder.
#
# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Import sdc package to build API Reference -------------------------------
import os

SDC_DOC_NO_API_REF_STR = 'SDC_DOC_NO_API_REF'

sdc_doc_no_api_ref = False  # Generate API Reference by default

if SDC_DOC_NO_API_REF_STR in os.environ:
    sdc_doc_no_api_ref = os.environ[SDC_DOC_NO_API_REF_STR] == '1'

if not sdc_doc_no_api_ref:
    try:
        import hpat  # TODO: Rename hpat module name to sdc
    except ImportError:
        print('IMPORT EXCEPTION: Cannot import hpat. ')
        print('Documentation generator for API Reference for a given module expects that module '
              'to be installed. Use conda/pip install hpat to install it prior to using API Reference generation')
        print('If you want to disable API Reference generation, set the environment variable SDC_DOC_NO_API_REF=1')

        raise

# -- Project information -----------------------------------------------------

project = 'Intel® Scalable Dataframe Compiler'
copyright = '2019, Intel Corporation'
author = 'Intel Corporation'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ----------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.extlinks',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.graphviz',
    'sphinx.ext.coverage'
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sdc-sphinx-theme'

html_theme_path = ['.']

html_theme_options = {
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_sidebars = {
   '**': ['globaltoc.html', 'sourcelink.html', 'searchbox.html', 'relations.html'],
 }

html_show_sourcelink = False

# -- Todo extension configuration  ----------------------------------------------
todo_include_todos = True
todo_link_only = False

# -- InterSphinx configuration: looks for objects in external projects -----
# Add here external classes you want to link from Intel SDC documentation
# Each entry of the dictionary has the following format:
#      'class name': ('link to object.inv file for that class', None)
intersphinx_mapping = {
    'pandas.Series': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'numpy.array': ('https://docs.scipy.org/doc/numpy', None),
}

# -- Napoleon extension configuration (Numpy and Google docstring options) -------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True


# -- Auto-section label configuration -----------------------------------------------
autosectionlabel_prefix_document = True


# -- Autodoc configuration ----------------------------------------------------------
autodoc_docstring_signature = True


# -- Auto-summary configuration -----------------------------------------------------
autosummary_generate = True

# -- Prepend module name to an object name or not -----------------------------------
add_module_names = False


"""
import API_Doc

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'HPATdoc'

# -- Options for LaTeX output ---------------------------------------------

latex_engine = 'pdflatex'
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'HPAT.tex', 'HPAT Documentation',
     'Intel', 'manual'),
]

pdf_documents = [
    ('index', u'HPATDocumentation', u'HPAT Documentation', u'Rujal Desai'),
]

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'hpat', 'HPAT Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'HPAT', 'HPAT Documentation',
     author, 'HPAT', 'One line description of project.',
     'Miscellaneous'),
]

numfig = True

# configuration for intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'numba': ('https://numba.pydata.org/numba-doc/dev', None),
    'numpy': ('http://docs.scipy.org/doc/numpy', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
}
"""
