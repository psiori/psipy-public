# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXDIR     = ./dist
BUILDDIR      = _build
OUTPUTDIR     = ./dist/output
SOURCES       = ../psipy

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help $(SOURCEDIR) $(BUILDDIR) $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ $(SOURCEDIR) $(BUILDDIR) $(SPHINXOPTS) $(O)

init:    # create directory for the sphinx documentation and copy the index.rst into it
	mkdir -p $(SPHINXDIR)
	cp *.rst $(SPHINXDIR)
	cp ../*.md $(SPHINXDIR)

SHELL:=/bin/bash

apidoc:  # write api files according to which sphinx will build the actual documentation
	sphinx-apidoc --module-first --separate -f -o $(SPHINXDIR)/source $(SOURCES)
	#--no-headings --separate
	# sphinx-apidoc doesn't allow setting maxdepth on subpackages
	echo $(shell pwd)
	#lf=$'\n';
	# sed -i.bak -e 's/.. toctree::/.. toctree::\'$$'\n''   :maxdepth: 1/g' -- ${SPHINXDIR}/source/*
	# sphinx toctree is very indentation sensitive, make it uniform
	# sed -i'' "s/    /   /g" -- "${SPHINXDIR}/source/*"

html:    # build a documentation in html format
	sphinx-build -b html -j auto -c . $(SPHINXDIR) $(OUTPUTDIR)

open:    # open the documentation
	open $(OUTPUTDIR)/index.html

rm:      # delete the sphinx-documentation folder recursively
	rm -rf $(SPHINXDIR)

all: rm
	make init
	make apidoc
	make html
	make open

latex:   # experimental, build latex files, see further instructions in output
	sphinx-build -b latex -j auto -c ./ $(SPHINXDIR) $(OUTPUTDIR)
