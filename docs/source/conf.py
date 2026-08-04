import datetime
import os.path as osp
import sys

from sphinx.application import Sphinx

# Add JraphX to path for autodoc
sys.path.insert(0, osp.abspath("../../src"))

import jraphx

version = jraphx.__version__

author = "JraphX Contributors"
project = "jraphx"
copyright = f"{datetime.datetime.now().year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

# Use Read the Docs theme
html_theme = "sphinx_rtd_theme"
# html_logo = None
# html_favicon = None
html_static_path = ["_static"]
templates_path = ["_templates"]

add_module_names = False
autodoc_member_order = "bysource"

intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    # 'numpy': ('http://docs.scipy.org/doc/numpy', None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/stable", None),
}

typehints_use_rtype = False
typehints_defaults = "comma"


def setup(app: Sphinx) -> None:
    r"""Configure the Sphinx application.

    Args:
        app: The Sphinx application being built.
    """
    # Keep type hints in the rendered signatures.
    if "autodoc-process-signature" in app.events.listeners:
        del app.events.listeners["autodoc-process-signature"]
