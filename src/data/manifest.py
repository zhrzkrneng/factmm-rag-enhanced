"""Dataset manifest generation.

Responsibility: build and serialize a dataset manifest (using the schema
defined in src/common/manifest.py) for a given data split — recording
patient IDs, study IDs, relative image paths, and record counts, but
never raw report text or images, so manifests are safe to inspect and
version-control. No implementation yet.
"""
