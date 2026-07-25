"""Abstract retriever interface.

Responsibility: define the common interface (encode query, encode
candidate, score/relevance function) that every concrete retriever —
baseline or innovation — must implement, so downstream code (indexing,
evaluation, RAG dataset construction) can operate on any retriever
polymorphically. No implementation yet.
"""
