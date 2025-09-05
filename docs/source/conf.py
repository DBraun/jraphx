import datetime
import os.path as osp
import sys

# Add JraphX to path for autodoc
sys.path.insert(0, osp.abspath("../../src"))

try:
    import jraphx

    version = jraphx.__version__
except ImportError:
    version = "0.0.1"

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

suppress_warnings = ["autodoc.import_object"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    # 'numpy': ('http://docs.scipy.org/doc/numpy', None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/stable", None),
}

typehints_use_rtype = False
typehints_defaults = "comma"

# Remove thumbnails for tutorials that no longer exist
# nbsphinx_thumbnails = {}


def setup(app):
    r"""Setup sphinx application."""
    # Remove Jinja templating since we don't use it anymore
    # app.connect('source-read', rst_jinja_render)

    # Remove version alert JS since it's PyG-specific
    # app.add_js_file('js/version_alert.js')

    # Do not drop type hints in signatures:
    try:
        del app.events.listeners["autodoc-process-signature"]
    except KeyError:
        pass  # May not exist in all Sphinx versions
