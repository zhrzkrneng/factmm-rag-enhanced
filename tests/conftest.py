"""Pytest configuration: make the project's ``src`` package importable.

This project has no ``pyproject.toml``/``setup.py`` installable package
yet, so pytest's default rootdir discovery won't put the repository root
on ``sys.path`` automatically. This conftest does that once, for every
test in the suite, so tests can simply ``import src...`` the same way
the Colab notebook cells do after Cell 06.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
