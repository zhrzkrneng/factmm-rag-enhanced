"""Dataset integrity checks.

Responsibility: detect missing image files, missing/empty finding or
impression text, duplicate (patient_id, study_id) pairs within a split,
and studies where a frontal view cannot be identified, per the integrity
checks required in docs/data_requirements.md. Raises loudly on failure
rather than silently skipping bad records. No implementation yet.
"""
