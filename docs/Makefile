# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = -j1
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = HPAT
SOURCEDIR     = .
BUILDDIR      = _build
DEVBUILDDIR   = _builddev
PYRUN         = python
DOCSRCDIR     = ./API_Doc

# User-friendly check for sphinx-build
ifeq ($(shell which $(SPHINXBUILD) >/dev/null 2>&1; echo $$?), 1)
$(error The '$(SPHINXBUILD)' command was not found. Make sure you have Sphinx installed)
endif


ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(SPHINXOPTS)
DEV_ALLSPHINXOPTS = -d $(DEVBUILDDIR)/doctrees $(SPHINXOPTS)

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	

clean:
		rm -rf $(BUILDDIR)/* $(DEVBUILDDIR)/*	  
		rm -rf $(SOURCEDIR)$(DOCSRCDIR)
		rm -rf $(SOURCEDIR)/usersource/api/*

.PHONY: help Makefile

html:
	rm -rf $(SOURCEDIR)$(DOCSRCDIR)
	$(PYRUN) rename_function.py
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) usersource -t user $(BUILDDIR)/html
	$(PYRUN) CleanRSTfiles.py
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) usersource -t user $(BUILDDIR)/html 
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."  
     

developerhtml:
	$(SPHINXBUILD) -b html $(DEV_ALLSPHINXOPTS) devsource -t developer $(DEVBUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(DEVBUILDDIR)/html."     

latex:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) usersource $(BUILDDIR)/latex
	@echo
	@echo "Build finished; the LaTeX files are in $(BUILDDIR)/latex."
	@echo "Run \`make all-pdf' in that directory to run these through (pdf)latex"

devlatex:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) devsource $(DEVBUILDDIR)/latex
	@echo
	@echo "Build finished; the LaTeX files are in $(DEVBUILDDIR)/latex."
	@echo "Run \`make all-pdf' in that directory to run these through (pdf)latex"
