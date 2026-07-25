"""Cross-cutting utilities shared by every other package in this project.

Responsibility: deterministic seeding, logging configuration, config
file loading/validation, the dataset manifest schema, and shared
exception types. Contains no dataset-specific, retrieval-specific, or
generation-specific logic — anything domain-specific belongs in
src/data, src/retrieval, src/generation, src/baseline, or src/innovation
instead.
"""
