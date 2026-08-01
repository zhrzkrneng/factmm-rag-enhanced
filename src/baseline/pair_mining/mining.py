"""Top-k positive pair mining.

Responsibility: apply the configurable chexbert_threshold,
radgraph_threshold, and top_k parameters to select positive report pairs
per query, with provenance (scores + which threshold each pair passed)
recorded, and leakage prevention across dataset splits. Must expose
top_k as a config value rather than hard-coding either the paper's
stated value (2) or the shipped scripts' default (3) — see
docs/risk_register.md #1b. No implementation yet.
"""
