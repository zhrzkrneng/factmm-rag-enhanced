# Reproduction Matrix

Status legend: `NOT_STARTED` / `PLANNED` / `IN_PROGRESS` / `IMPLEMENTED` /
`TESTED` / `BLOCKED`. Nothing below is `IMPLEMENTED` yet — Phase 0 is
documentation-only, no model code has been written.

Confidence reflects how well-specified the component is from currently
available sources (official repo only; paper PDF pending).

| Component | Paper description | Official-code location | Required inputs | Expected outputs | Status | Confidence | Open questions |
|---|---|---|---|---|---|---|---|
| Data parsing | UNKNOWN (pending PDF) | `data/parse.py` | Per-line `.tok` image-path / finding / impression files | `{image, finding, impression}` JSON records | NOT_STARTED | High (code fully read) | Are line-aligned `.tok` files the paper's actual preprocessing, or a MIMIC-CXR-community convention? |
| Frontal-view selection | UNKNOWN | Implicit: `data.py` always uses `image[0]` | Multi-image study records | Single primary image per study | NOT_STARTED | Medium | Is `image[0]` guaranteed to be frontal upstream, or does this need explicit view-position filtering from MIMIC-CXR metadata? |
| Patient/study split | UNKNOWN | Not explicit in code; MIMIC-CXR's own train/valid/test split is assumed inherited | Dataset with patient/study identifiers | Patient-disjoint train/valid/test manifests | NOT_STARTED | Low-Medium | Official code trusts an externally-provided split; this project must independently verify no patient leakage rather than assume it |
| RadGraph annotation | UNKNOWN | `data/label.py` | Report `finding` text | Entities/relations JSON | NOT_STARTED | Medium | Exact RadGraph model/checkpoint version; GPU requirement for CheXbert inference |
| CheXbert labeling | UNKNOWN | `data/label.py` | Report `finding` text | 14-class labels, 5-class subset kept | NOT_STARTED | High | Why exactly these 5 classes (Cardiomegaly, Edema, Consolidation, Atelectasis, Pleural Effusion)? |
| Factual similarity scoring | UNKNOWN | `gen_similarity.py` | Labeled reports | All-pairs CheXbert + RadGraph similarity matrices | NOT_STARTED | High | Compute cost at full corpus scale (125,417²) — needs chunking/sampling strategy for a small-scale MVP |
| Top-k positive mining | UNKNOWN | `gen_topk_pos.py`, `merge_topk_pos.py` | Similarity matrices | Per-query positive candidate ID lists | NOT_STARTED | High | Confirmed thresholds: chex≥1.0, radg≥0.4, k=3 |
| Retriever architecture | UNKNOWN | `src/retriever/DPR/multi_model.py` | CLIP ViT-B/32 + T5 (`t5-ance`/MARVEL) | Query/candidate embeddings | NOT_STARTED | High | MARVEL checkpoint availability/license on HuggingFace |
| Retriever training (DPR stage) | UNKNOWN | `src/retriever/DPR/train.py` | Train/valid JSON + mined positives | Trained checkpoint | NOT_STARTED | High | GPU memory footprint at batch size 32 with full images |
| Hard-negative mining | UNKNOWN | `gen_hard_negatives.py` | DPR embeddings, labeled reports | Per-query hard-negative ID lists | NOT_STARTED | High | — |
| Retriever training (ANCE stage) | UNKNOWN | `src/retriever/ANCE/train.py` | DPR checkpoint + hard negatives | Trained checkpoint | NOT_STARTED | Medium (file not yet fully read) | Exact loss formulation vs. DPR stage |
| Embedding export / indexing | UNKNOWN | `gen_embeddings.py`, FAISS index code | Trained retriever | Corpus embedding matrix + FAISS index | NOT_STARTED | Medium | IVFFlat vs. FlatIP choice at scale |
| Retrieval metrics | UNKNOWN | `evaluate_retriever.py` | Embeddings + positive-matrix | MRR/Recall/NDCG @ multiple K | NOT_STARTED | High | — |
| RAG dataset construction | UNKNOWN | `build_rag_dataset.py` | KNN index + query/corpus JSON | LLaVA-format prompts with 1 retrieved report | NOT_STARTED | High | Confirmed patient/study leakage filter exists; must reproduce exactly |
| Generator (LLaVA) | UNKNOWN | `install_llava.sh`, `train_llava.sh`, `inference_llava.sh` | Retrieved-augmented dataset, LLaVA base weights | Generated reports | NOT_STARTED | Low | README marks generator training code as **not released** at clone time; treat scripts as unverified until run |
| Non-RAG baseline (VQA) | UNKNOWN | `build_nonrag_dataset.py`, `vqa/*.sh` | Query JSON only | Generated reports, no retrieval | NOT_STARTED | Low-Medium | Not yet fully read |
| Generation metrics | UNKNOWN | `src/evaluation.py` | Reference + predicted texts | F1RadGraph, F1CheXbert, ROUGE-L, BLEU-4, BERTScore | NOT_STARTED | High | — |
| Bootstrap CI / significance testing | Not in official code (project requirement) | N/A | Per-example metric scores | Confidence intervals, p-values | NOT_STARTED | N/A (`PROPOSED_EXTENSION`) | Method choice (paired bootstrap vs. permutation test) — our decision, not the paper's |
| Adaptive multi-report retrieval (Innovation A) | Not in paper/official code | N/A | Trained retriever + confidence signal | Variable-k retrieval | NOT_STARTED | N/A (`PROPOSED_EXTENSION`) | — |
| Fact-aware reranking (Innovation B) | Not in paper/official code | N/A | Retrieved candidates + factual scores | Reranked candidates | NOT_STARTED | N/A (`PROPOSED_EXTENSION`) | — |
| Multi-report fusion (Innovation C) | Not in paper/official code | N/A | Multiple retrieved reports | Structured evidence context | NOT_STARTED | N/A (`PROPOSED_EXTENSION`) | — |
| Confidence gating (Innovation D) | Not in paper/official code | N/A | Retrieval confidence signal | Retrieve/no-retrieve decision | NOT_STARTED | N/A (`PROPOSED_EXTENSION`) | — |
| Uncertainty/contradiction detection (Innovation E) | Not in paper/official code | N/A | Retrieved facts + generated report | Warnings/confidence summary | NOT_STARTED | N/A (`PROPOSED_EXTENSION`) | — |
| Longitudinal context (Innovation F) | Not in paper/official code | N/A | Prior-study data (if available) | Longitudinal-aware generation | NOT_STARTED | N/A (`PROPOSED_EXTENSION`, optional) | Depends on legal/structural availability of prior studies |

All "Paper description" cells read `UNKNOWN` because `paper/factmm_rag.pdf`
has not been supplied. This column will be revised once the PDF is
available, and every row's confidence/status will be re-evaluated against
paper text, not just code.
