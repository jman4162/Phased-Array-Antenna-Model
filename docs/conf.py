# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the project root to the path so autodoc can find the package
sys.path.insert(0, os.path.abspath(".."))

import phased_array

# -- Project information -----------------------------------------------------
project = "Phased Array Modeling"
copyright = "2024, John Hodge"
author = "John Hodge"
version = phased_array.__version__
release = phased_array.__version__

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Suppress duplicate warnings for dataclass attributes
suppress_warnings = ["autodoc.duplicate_object"]

# -- Options for Napoleon (NumPy docstrings) ---------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "plotly": ("https://plotly.com/python-api-reference/", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "Phased Array Modeling"
html_short_title = "phased-array"
html_logo = None
html_favicon = None

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/jman4162/Phased-Array-Antenna-Model",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/phased-array-modeling/",
            "icon": "fa-solid fa-box",
            "type": "fontawesome",
        },
    ],
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "show_nav_level": 2,
    "navbar_align": "left",
    "navbar_center": ["navbar-nav"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}

html_context = {
    "github_user": "jman4162",
    "github_repo": "Phased-Array-Antenna-Model",
    "github_version": "main",
    "doc_path": "docs",
}

# -- Options for copybutton --------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_remove_prompts = True

# -- Options for doctest -----------------------------------------------------
doctest_global_setup = """
import numpy as np
import phased_array as pa
"""

# -- Options for MathJax -----------------------------------------------------
mathjax3_config = {
    "tex": {
        "macros": {
            "vec": [r"\mathbf{#1}", 1],
            "mat": [r"\mathbf{#1}", 1],
        }
    }
}
