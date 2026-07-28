"""Sphinx configuration for the Master Thesis Template."""

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "Master Thesis Template"
copyright = "2025, IAS University of Stuttgart"
author = "IAS"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_nb",
]

# MyST parser
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
]
myst_heading_anchors = 3

# Notebook execution (off = use pre-computed outputs)
nb_execution_mode = "off"

# Autodoc
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_mock_imports = [
    "torch",
    "torch_geometric",
    "pytorch_lightning",
    "lightning",
    "mlflow",
    "hydra",
    "omegaconf",
    "pandas",
    "matplotlib",
    "numpy",
    "scienceplots",
    "point_cloud_utils",
    "ddacs",
    "pyvista",
]

# Napoleon (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# Theme
html_theme = "furo"
html_title = "Master Thesis Template"
html_theme_options = {
    "navigation_with_keys": True,
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]
html_static_path = ["_static"]

# Keep plain ASCII punctuation (no automatic en/em dashes or curly quotes)
smartquotes = False
