"""Baseline hyperparameter documentation and defaults.

Responsibility: centralize the baseline reproduction's default
hyperparameters as sourced from configs/baseline/, and document the two
known paper-vs-shipped-code discrepancies (top_k positives per query:
paper=2 vs. code=3; contrastive temperature: paper fixed tau=0.01 vs.
code's learned logit_scale) so both variants can be run and compared
rather than one being silently assumed correct. See
docs/risk_register.md #1b and #1c. No implementation yet.
"""
