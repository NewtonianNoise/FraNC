# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "FraNC"
copyright = "2025, Tim Kuhlbusch"
author = "Tim Kuhlbusch"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "autoapi.extension",
]

autoapi_dirs = ["../../src"]

templates_path = ["_templates"]
exclude_patterns: list = []

autodoc_typehints = "description"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "nature"
html_static_path = ["_static"]
html_theme_options = {
    "sidebarwidth": "400px",
}
html_sidebars = {
    "**": ["globaltoc.html", "localtoc.html", "sourcelink.html", "searchbox.html"]
}
