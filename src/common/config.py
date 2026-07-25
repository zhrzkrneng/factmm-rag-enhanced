"""Configuration loading and validation.

Responsibility: load YAML/JSON configuration files from configs/ into
typed, validated Python objects, so that no experiment hyperparameter is
ever hard-coded in source (per CLAUDE.md's "use configuration files
instead of hard-coded values" rule). Also responsible for surfacing clear,
early validation errors when a config is missing required fields or has
values outside an expected range. No implementation yet.
"""
