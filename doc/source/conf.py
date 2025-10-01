# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import franc

project = "FraNC"
copyright = "2025, Tim J. Kuhlbusch et al."
author = "Tim J. Kuhlbusch et al."
version = franc.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "autoapi.extension",
]

autoapi_dirs = ["../../src"]

templates_path = ["_templates"]
exclude_patterns: list = []

autodoc_typehints = "description"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_static_path = ["_static"]
html_theme_options = {
    "sidebarwidth": "400px",
    "navigation_depth": 6,
}
html_sidebars = {"**": ["globaltoc.html", "sourcelink.html", "searchbox.html"]}
