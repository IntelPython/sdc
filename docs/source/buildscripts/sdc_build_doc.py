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

import os
import sys
from setuptools import Command


# Sphinx User's Documentation Build

class SDCBuildDoc(Command):
    description = 'Builds Intel SDC documentation'
    user_options = [
        ('format', 'f', 'Documentation output format (html, pdf)'),
        ('doctype', 't', 'Documentation type (user, dev)'),
    ]

    @staticmethod
    def _remove_cwd_from_syspath():
        # Remove current working directory from PYTHONPATH to avoid importing SDC from sources vs.
        # from installed SDC package
        cwd = ['', '.', './', os.getcwd()]
        sys.path = [p for p in sys.path if p not in cwd]

    def initialize_options(self):
        self.format = 'html'
        self.doctype = 'user'

    def finalize_options(self):
        pass

    def run(self):
        self.sdc_build_doc_command.finalize_options()
        self.sdc_build_doc_command.run()

    def __init__(self, dist):
        super(SDCBuildDoc, self).__init__(dist)
        self.format = 'html'
        self.doctype = 'user'
        try:
            from sphinx.setup_command import BuildDoc
        except ImportError:
            raise ImportError('Cannot import Sphinx. '
                              'Sphinx is the expected dependency for Intel SDC documentation build')

        self._remove_cwd_from_syspath()
        self.sdc_build_doc_command = BuildDoc(dist)
        self.sdc_build_doc_command.initialize_options()


# Sphinx Developer's Documentation Build

#class build_devdoc(build.build):
#    description = "Build developer's documentation"
#
#    def run(self):
#        spawn(['rm', '-rf', 'docs/_builddev'])
#        spawn(['sphinx-build', '-b', 'html', '-d', 'docs/_builddev/docstrees',
#               '-j1', 'docs/devsource', '-t', 'developer', 'docs/_builddev/html'])
