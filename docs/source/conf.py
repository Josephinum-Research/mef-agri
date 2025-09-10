# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.append(
    os.path.join(os.path.split(__file__)[0], os.pardir, os.pardir)
)
print(sys.path[-1])

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'mef-agri - a model evaluation framework for agricultural models'
copyright = '2025, Josephinum Research'
author = 'Andreas Ettlinger'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc'
]

templates_path = ['_templates']
exclude_patterns = []
autodoc_typehints = 'none'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'haiku'
html_theme_options = {
#    'nosidebar': False,  # nature theme
#    'sidebarwidth': '25%'  # natur theme
    'full_logo': True,
    'headingcolor': 'Black',
    'linkcolor': 'MidnightBlue',
    'visitedlinkcolor': 'MidnightBlue',
    'hoverlinkcolor': 'ForestGreen'
}
html_static_path = ['_static']
html_logo = os.path.join(os.pardir, '__imgs__', '2014_JR.jpg')
