"""Logging configuration utilities.

Responsibility: provide a single, project-wide function to configure
Python's `logging` module (format, level, handlers) consistently across
scripts, notebooks, and Colab cells, ensuring log output never includes
raw patient data, report text, or file paths that could leak PHI (per
CLAUDE.md's "never include private medical data in logs" rule). No
implementation yet.
"""
