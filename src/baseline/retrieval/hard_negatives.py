"""Hard negative mining for the ANCE-style training stage.

Responsibility: for each query, retrieve top-N candidates via the
current retriever's FAISS index, keep only candidates confirmed
factually dissimilar (chexbert_similarity < threshold AND
radgraph_similarity < threshold), and select the hardest (lowest-scoring)
of those as hard negatives, matching official FactMM-RAG
gen_hard_negatives.py exactly. No implementation yet.
"""
