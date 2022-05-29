# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config
#
# For ReStructuredText primer see:
# http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
#
# For numpy docstring guide see:
# https://numpydoc.readthedocs.io/en/latest/format.html#sections
#
# For bibtex referencing see:
# https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import datetime

# Add climopy to path for sphinx-automodapi
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

_today = datetime.datetime.today()
project = 'ClimoPy'
copyright = f'{_today.year}, Luke L. B. Davis'
author = 'Luke L. B. Davis'

# The short X.Y version
version = ''

# The full version, including alpha/beta/rc tags
release = ''


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',  # >>> examples
    'sphinx.ext.extlinks',  # for :pr:, :issue:, :commit:
    'sphinx.ext.autosectionlabel',  # use :ref:`Heading` for any heading
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',  # for NumPy style docstrings, instead of reStructred Text
    'sphinx.ext.intersphinx',  # external links
    'sphinx_automodapi.automodapi',  # see: https://github.com/lukelbd/sphinx-automodapi/tree/proplot-mods  # noqa: #501
    'sphinxcontrib.bibtex',  # see: https://sphinxcontrib-bibtex.readthedocs.io/en/latest/quickstart.html  # noqa: #501
    'sphinx_copybutton',
    'sphinx-rtd-light-dark',
]

extlinks = {
    'issue': ('https://github.com/lukelbd/climopy/issues/%s', 'GH#'),
    'commit': ('https://github.com/lukelbd/climopy/commit/%s', '@'),
    'pr': ('https://github.com/lukelbd/climopy/pull/%s', 'GH#'),
}

# Cupybutton configuration
# See: https://sphinx-copybutton.readthedocs.io/en/latest/
copybutton_prompt_text = r'>>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: '
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True
copybutton_remove_prompts = True

# Bibtex file locations
import glob
bibtex_bibfiles = glob.glob('_bibfiles/*')

# Run doctest test blocks
doctest_test_doctest_blocks = 'default'

# Generate stub pages whenever ::autosummary directive encountered
# This way don't have to call sphinx-autogen manually
autosummary_generate = True

# Use automodapi tool, created by astropy people
# See: https://sphinx-automodapi.readthedocs.io/en/latest/automodapi.html#overview
# Normally have to *enumerate* function names manually. This will document
# them automatically. Just be careful, if you use from x import *, to exclude
# them in the automodapi:: directive
automodapi_toctreedirnm = 'api'  # create much better URL for the page
automodsumm_inherited_members = False

# Turn off code and image links for embedded mpl plots
# plot_html_show_source_link = False
# plot_html_show_formats = False

# One of 'class', 'both', or 'init'
# The 'both' concatenates class and __init__ docstring
# See: http://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
autoclass_content = 'both'

# Set up mapping for other projects' docs
intersphinx_mapping = {
    'matplotlib': ('https://matplotlib.org', None),
    'pint': ('https://pint.readthedocs.io/en/stable/', None),
    'cf_xarray': ('https://cf-xarray.readthedocs.io/en/stable/', None),
    'sphinx': ('http://www.sphinx-doc.org/en/stable/', None),
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'xarray': ('http://xarray.pydata.org/en/stable', None),
    'proplot': ('http://proplot.readthedocs.io/en/latest', None),
}

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False  # confusing, because I use submodules for *organization*

# Fix duplicate class member documentation from autosummary + numpydoc
# See: https://github.com/phn/pytpm/issues/3#issuecomment-12133978
numpydoc_show_class_members = False

# Napoleon options
# See: http://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
napoleon_use_ivar = False
napoleon_use_param = False
napoleon_use_keyword = False
napoleon_use_rtype = False
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = False  # move init doc to 'class' doc
napoleon_preprocess_types = True
napoleon_type_aliases = {
    # Python or inherited terms
    # NOTE: built-in types are automatically included
    'sequence': ':term:`sequence`',
    'callable': ':py:func:`callable`',
    'ndarray': ':py:class:`numpy.ndarray`',
    'dict-like': ':term:`dict-like <mapping>`',
    'path-like': ':term:`path-like <path-like object>`',
    'array-like': ':term:`array-like <array_like>`',
    # Climopy defined terms
    'unit-spec': ':py:func:`unit-spec <climopy.unit.ureg>`',
    'var-spec': ':py:func:`var-spec <climopy.cfvariable.vreg>`',
}

# The name of the Pygments (syntax highlighting) style to use.
# The light-dark theme toggler overloads this, but set default anyway
pygments_style = 'none'

# The master toctree document.
master_doc = 'index'

# The suffix(es) of source filenames.
source_suffix = '.rst'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = [
    'conf.py', 'sphinxext', '_build', '_templates', '_themes',
    '*.ipynb', '**.ipynb_checkpoints', '.DS_Store',
]

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
language = None

# Role. Default family is py, but can also set default role so don't need
# :func:`name`, :module:`name`, etc.
default_role = 'py:obj'

# -- Options for HTML output -------------------------------------------------

# Logo
html_logo = os.path.join('_static', 'logo_square.png')

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_light_dark'
html_theme_options = {
    'logo_only': True,
    'display_version': False,
    'collapse_navigation': True,
    'navigation_depth': 4,
    'prev_next_buttons_location': 'bottom',  # top and bottom
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large. Static folder is for CSS and image files.
# For icons see: https://icons8.com/icon
html_favicon = os.path.join('_static', 'logo_blank.ico')

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'climopydoc'


# -- Options for LaTeX output ------------------------------------------------

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
    (master_doc, 'climopy.tex', 'ClimoPy Documentation',
     'Luke L. B. Davis', 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (
        master_doc,
        'climopy',
        'ClimoPy Documentation',
        [author],
        1
    )
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        'climopy',
        'ClimoPy Documentation',
        author,
        'climopy',
        'Tools for working with climatological data.',
        'Miscellaneous'
    )
]


# -- Extension configuration -------------------------------------------------

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
