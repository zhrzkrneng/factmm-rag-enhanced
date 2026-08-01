"""Result aggregation and reporting.

Responsibility: collect metric outputs (retrieval + generation +
significance) for an experiment run into a single machine-readable
JSON/CSV result record, alongside seed, config, code commit, dataset
manifest reference, package versions, and hardware/runtime metadata, per
CLAUDE.md's experiment-recording rules. Also builds comparison tables
against the paper's reported Table 1 numbers. No implementation yet.
"""
