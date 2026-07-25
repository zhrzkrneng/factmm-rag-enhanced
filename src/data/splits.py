"""Patient-level split validation.

Responsibility: independently verify — never assume — that no
patient_id appears in more than one of {train, valid, test}, and that
retrieval corpora used during training/evaluation never leak a query's
own study or patient. This is a stricter, standalone check than the
official FactMM-RAG code's RAG-construction-time filter (see
docs/data_requirements.md). No implementation yet.
"""
