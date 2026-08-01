"""Innovation C: Diverse multi-report fusion.

Responsibility: when multiple reports are retrieved (via
src/innovation/adaptive_retrieval), extract factual units per report,
identify agreements vs. contradictions, remove redundant facts, preserve
clinically relevant negations, and produce a structured evidence object
(supporting facts, conflicting facts, source IDs, confidence scores) for
the generator, replacing raw text concatenation. Compared against raw
concatenation per docs/innovation_proposals.md.
"""
