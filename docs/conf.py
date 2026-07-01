"""Sphinx configuration for the Phenocoder documentation."""

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# Make the phenocoder package importable for autodoc.
sys.path.insert(0, str(_ROOT))


def _get_version() -> str:
    """Read ``__version__`` from the package without importing it.

    Importing ``phenocoder`` here would pull in heavy runtime deps (anndata,
    tensorflow, ...) that are not installed in the docs environment and are only
    mocked *during* autodoc, not at config-file execution time.
    """
    init = (_ROOT / "phenocoder" / "__init__.py").read_text()
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init, re.MULTILINE)
    return match.group(1) if match else "0.0.0"


__version__ = _get_version()

# -- Project information -----------------------------------------------------

project = "Phenocoder"
author = "devsystemslab"
copyright = f"2026, {author}"
release = __version__
version = __version__

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

# -- MyST -------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]
myst_heading_anchors = 3

# -- autodoc / autosummary --------------------------------------------------

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

# Heavy / optional runtime dependencies are mocked so the docs build without
# installing the full ML + spatial stack (this is the ReadTheDocs environment).
autodoc_mock_imports = [
    "tensorflow",
    "keras",
    "spatialdata",
    "spatialdata_plot",
    "squidpy",
    "scanpy",
    "anndata",
    "bbknn",
    "igraph",
    "leidenalg",
    "umap",
    "networkx",
    "skimage",
    "sklearn",
    "scipy",
    "joblib",
    "yaml",
    "tqdm",
    "matplotlib",
    "pandas",
    "numpy",
]

# -- napoleon (Google-style docstrings) -------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- intersphinx ------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "spatialdata": ("https://spatialdata.scverse.org/en/stable/", None),
}

# -- HTML output ------------------------------------------------------------

html_theme = "furo"
html_title = "Phenocoder"
html_static_path = ["_static"]
html_css_files = ["catppuccin.css"]

# Syntax highlighting: use neutral built-in Pygments styles that Furo already
# themes well; the Catppuccin palette is applied to both the chrome and the code
# blocks via docs/_static/catppuccin.css (Latte in light mode, Mocha in dark).
pygments_style = "default"
pygments_dark_style = "github-dark"

html_theme_options = {
    "light_logo": "logo.png",
    "dark_logo": "logo.png",
    "source_repository": "https://github.com/devsystemslab/phenocoder",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/devsystemslab/phenocoder",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 '
                "3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 "
                "0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-"
                ".82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 "
                "1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 "
                "0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82."
                "64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82"
                ".44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3."
                "65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46."
                '55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}
