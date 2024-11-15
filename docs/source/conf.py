# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
#
import sys

project = "gpax"
copyright = "2024, GPax authors"
author = "GPax authors"


sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))
import gpax

version = gpax.__version__
master_doc = "index"
language = "en"
latex_engine = "xelatex"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "myst_parser",
    "sphinx_toolbox.more_autodoc.typehints",
    "sphinx_autodoc_typehints",
    "autodoc2",
    # "sphinx.ext.napoleon",
    # "attr_utils.annotations",
    # "attr_utils.autoattrs",
]
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.10", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "numpyro": ("https://num.pyro.ai/en/stable", None),
    "markdown_it": ("https://markdown-it-py.readthedocs.io/en/latest", None),
}

# Autodoc settings ------------------------------------------------------------
autodoc2_packages = [
    {
        "path": "../../gpax",
        "exclude_files": ["_docs.py"],
    }
]
autodoc2_hidden_objects = ["dunder", "private", "inherited"]
autodoc2_class_docstring = "both"

templates_path = []
exclude_patterns = []

autodoc2_replace_annotations = [
    ("re.Pattern", "typing.Pattern"),
    ("markdown_it.MarkdownIt", "markdown_it.main.MarkdownIt"),
]
autodoc2_replace_bases = [
    (
        "sphinx.directives.SphinxDirective",
        "sphinx.util.docutils.SphinxDirective",
    ),
]
autodoc2_docstring_parser_regexes = [
    (r".*", "myst"),
]
# nitpicky = True
# nitpick_ignore_regex = [
#     (r"py:.*", r"docutils\..*"),
#     (r"py:.*", r"pygments\..*"),
#     (r"py:.*", r"typing\.Literal\[.*"),
# ]
# nitpick_ignore = [
#     ("py:obj", "myst_parser._docs._ConfigBase"),
#     ("py:exc", "MarkupError"),
#     ("py:class", "sphinx.util.typing.Inventory"),
#     ("py:class", "sphinx.writers.html.HTMLTranslator"),
#     ("py:obj", "sphinx.transforms.post_transforms.ReferencesResolver"),
# ]
#

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "alabaster"
# install via pip install sphinx-book-theme
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

# Myst settings ------------------------------------------------------------
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
    "linkify",
    "strikethrough",
    "substitution",
    "tasklist",
    "attrs_inline",
    "attrs_block",
]
