"""RAG dataset construction.

Responsibility: given a query set, a retriever's KNN results, and a
prompt template, build the fine-tuning/inference dataset for the
generator, explicitly filtering out same-study, same-patient, and
malformed (< 5 character) retrieved candidates — matching the official
FactMM-RAG build_rag_dataset.py and the paper's Appendix A.2 exactly. No
implementation yet.
"""
