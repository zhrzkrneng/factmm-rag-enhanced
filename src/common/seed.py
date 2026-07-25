"""Deterministic seed utilities.

Responsibility: provide a single, project-wide function to seed Python's
`random`, `numpy`, and `torch` random number generators from one integer
seed, so every experiment can record and reproduce the exact seed used
(per CLAUDE.md's "record random seeds" rule). No implementation yet.
"""
