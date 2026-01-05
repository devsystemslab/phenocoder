"""
Tests package initializer.

Making the `tests` directory a regular package allows test modules to import
`tests.conftest` (for example: `from tests.conftest import example_3d`).

Additionally, if the top-level `phenocoder` package is not importable because
the project wasn't installed in editable mode, this initializer will add the
project root to `sys.path` so tests run from the repository root can import
the package.
"""

from __future__ import annotations

import os
import sys
from typing import List

# Add repository root to sys.path to help tests import the top-level package
# when the project isn't installed in editable mode.
_repo_root = os.path.dirname(os.path.dirname(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Public API for the tests package (keeps this file explicit and minimal).
__all__: List[str] = []
