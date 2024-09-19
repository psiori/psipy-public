# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../"))


# -- Project information -----------------------------------------------------

project = "psipy-public"
copyright = "2019-2024, PSIORI GmbH"
author = "PSIORI GmbH"

primary_domain = "py"
highlight_language = "py"

# -- General configuration ---------------------------------------------------

master_doc = "index"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'Sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    # "autodocsumm",
    "recommonmark",
    "sphinx.ext.todo",
    # 'sphinx.ext.coverage',
    # 'sphinx.ext.ifconfig',
    # 'sphinx.ext.viewcode',
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.linkcode",
]

autodoc_default_options = {
    # "exclude-members": (
    #     "__weakref__,__module__,__dict__,__doc__,__abstractmethods__,_abc_impl,"
    #     "__annotations__"
    # ),
    "ignore-module-all": True,
    # "imported-members": False,
    # "inherited-members": True,
    "member-order": "groupwise",
    "members": True,
    "private-members": True,
    "show-inheritance": False,
    # "special-members": "__init__",
    "undoc-members": True,
    # "autosummary": True,
    # "autosummary-private-members": False,
    # "autosummary-undoc-members": False,
    # "autosummary-inherited-members": True,
    # # "autosummary-special-members": "",
    # # "autosummary-exclude-members": "",
    # "autosummary-imported-members": False,
    # "autosummary-ignore-module-all": False,
    # "autosummary-members": True,
    # "autosummary-no-nesting": False,
}


source_suffix = [".rst", ".md"]


always_document_param_types = True
autoclass_content = "both"
autodoc_inherit_docstrings = False
autodoc_member_order = "groupwise"
autodoc_mock_imports = ["gym", "ConfigSpace", "hpbandster"]
autodoc_typehints = "signature"

autosummary_generate = True
autosummary_imported_members = True

napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_keyword = False
napoleon_use_param = True
napoleon_use_rtype = True

todo_include_todos = True
typehints_document_rtype = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]


# Intersphinx
intersphinx_mapping = {
    "django": (
        "https://docs.djangoproject.com/en/dev/",
        "https://docs.djangoproject.com/en/dev/_objects/",
    ),
    "h5py": ("http://docs.h5py.org/en/latest/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "python": ("https://docs.python.org/3.6/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "Sphinx": ("http://www.sphinx-doc.org/en/master/", None),
    "zmq": ("https://pyzmq.readthedocs.io/en/latest/", None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Add html pages
# html_additional_pages = {"index": "index.rst"}

html_theme_options = {
    "canonical_url": "https://psipy-public.psiori.net",
    "logo_only": False,
    "display_version": False,
    "prev_next_buttons_location": "bottom",
    # "style_external_links": False,
    # "vcs_pageview_mode": "",
    # "style_nav_header_background": "white",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 10,
    "includehidden": True,
    "titles_only": False,
}


def autodoc_skip_member_handler(app, what, name, obj, skip, options):
    if name.startswith("test_") or name == "__init__" or name.startswith("_abc_"):
        return True
    return skip


# Automatically called by sphinx at startup
def setup(app):
    # Connect the autodoc-skip-member event from apidoc to the callback
    app.connect("autodoc-skip-member", autodoc_skip_member_handler)
    app.add_css_file("css/hatnotes.css")
    app.add_css_file("css/custom.css")


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None
    filename = info["module"].replace(".", "/")
    return "https://github.com/psiori/psipy-public/tree/develop/%s.py" % filename
