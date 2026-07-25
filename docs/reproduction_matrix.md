# Reproduction Matrix

Status legend: `NOT_STARTED` / `PLANNED` / `IN_PROGRESS` / `IMPLEMENTED` /
`TESTED` / `BLOCKED`. Nothing below is `IMPLEMENTED` yet — Phase 0 is
documentation-only, no model code has been written.

Now that `paper/factmm_rag.pdf` has been read (see `docs/paper_analysis.md`
for full detail), most rows below have a real "Paper description" and
higher confidence than the original code-only pass. Two genuine
paper-vs-code discrepancies were found and are called out explicitly
rather than silently resolved (see the `top_k` and `τ` rows).

| Component | Paper description | Official-code location | Required inputs | Expected outputs | Status | Confidence | Open questions |
|---|---|---|---|---|---|---|---|
| Data parsing | Concatenate finding+impression as report text; select frontal view per study (§3.2 impl. details) | `data/parse.py` | Per-line `.tok` image-path / finding / impression files | `{image, finding, impression}` JSON records | NOT_STARTED | High | None — matches code |
| Frontal-view selection | Explicitly stated: "we select the frontal view" | Implicit: `data.py` always uses `image[0]` | Multi-image study records | Single primary image per study | NOT_STARTED | High (resolved by paper) | None |
| Patient/study split | Uses MIMIC-CXR's standard processed split (125,417/991/1,624 per Delbrouck et al. 2023) | Not explicit in code; split inherited externally | Dataset with patient/study identifiers | Patient-disjoint train/valid/test manifests | NOT_STARTED | Medium | Paper doesn't independently verify no leakage in that external split — this project still audits it directly |
| RadGraph annotation | RadGraph extracts entities+relations (no version specified) | `data/label.py`, pinned `radgraph==0.0.9` | Report `finding` text | Entities/relations JSON | NOT_STARTED | Medium | Exact model version behind paper's numbers vs. package `0.0.9` |
| CheXbert labeling | 5-class subset used for both mining and evaluation (Cardiomegaly, Edema, Consolidation, Atelectasis, Pleural Effusion) | `data/label.py` | Report `finding` text | 14-class labels, 5-class subset kept | NOT_STARTED | High | None |
| Factual similarity scoring | Eq.1: RadGraph-based F1-style overlap `s(q,d)=2|q̂∩d̂|/(len(q̂)+len(d̂))`, restricted first to same-label candidates | `gen_similarity.py` | Labeled reports | All-pairs CheXbert + RadGraph similarity matrices | NOT_STARTED | High | Compute cost at 125,417² scale needs a small-scale substitute for the MVP |
| Top-k positive mining | Paper: **top 2** pairs/query, threshold δ (Fig. 2 shows chex=1.0 best) | `gen_topk_pos.py` (shipped scripts use **top_k=3**) | Similarity matrices | Per-query positive candidate ID lists | NOT_STARTED | High, but **discrepancy**: paper says top-2, code default is top-3 — make `top_k` configurable and run both |
| Retriever architecture | MARVEL (T5-ANCE text + ViT vision), two init checkpoints (WebQA/ClueWeb) compared | `src/retriever/DPR/multi_model.py` | CLIP ViT-B/32 + T5 (`t5-ance`/MARVEL) | Query/candidate embeddings | NOT_STARTED | High | MARVEL checkpoint availability/license on HuggingFace |
| Retriever training (stage 1) | AdamW, epochs=15, early-stop=5, batch=32, lr=5e-6, **τ=0.01** (fixed) | `src/retriever/DPR/train.py` (uses a **learned** `logit_scale`, not a fixed τ) | Train/valid JSON + mined positives | Trained checkpoint | NOT_STARTED | High, but **discrepancy**: paper states a fixed temperature, code learns `logit_scale` — document both, don't silently pick one |
| Hard-negative mining | "Modality-balanced hard negatives" (method cited from MARVEL/T5-ANCE papers, no thresholds restated) | `gen_hard_negatives.py` | DPR embeddings, labeled reports | Per-query hard-negative ID lists | NOT_STARTED | Medium (thresholds only from code) | Whether code's `chex<1.0 AND radg<0.4`, `topN=100`, `keep=2` exactly match the cited MARVEL recipe |
| Retriever training (stage 2 / ANCE) | Same as above, "after in-batch negative training" | `src/retriever/ANCE/train.py` | DPR checkpoint + hard negatives | Trained checkpoint | NOT_STARTED | Medium | Exact loss formulation vs. stage 1, beyond what's stated |
| Embedding export / indexing | Full corpus retrieval (Figure 1, stage 3) | `gen_embeddings.py`, FAISS index code | Trained retriever | Corpus embedding matrix + FAISS index | NOT_STARTED | Medium | IVFFlat vs. FlatIP choice at scale (not paper-specified) |
| Retrieval metrics | MRR used as an intermediate/analysis metric (Fig. 3, 4), not in main Table 1 | `evaluate_retriever.py` | Embeddings + positive-matrix | MRR/Recall/NDCG @ multiple K | NOT_STARTED | High | Paper's Table 2 "retrieval-only" setting uses generation metrics (F1CheXbert etc.), not MRR/Recall — two different retrieval-evaluation modes to reproduce |
| RAG dataset construction | Filters self-report/self-patient/short (<5 char) results (Appendix A.2) | `build_rag_dataset.py` | KNN index + query/corpus JSON | LLaVA-format prompts with 1 retrieved report | NOT_STARTED | High | None — paper and code agree exactly |
| Generator (LLaVA) | LLaVA-1.5 from vicuna-7b-v1.5, epochs=1, lr=2e-5, batch=128, **8x A6000, ~4h** | `install_llava.sh`, `train_llava.sh`, `inference_llava.sh` | Retrieved-augmented dataset, LLaVA base weights | Generated reports | NOT_STARTED | Medium — hyperparameters now confirmed by paper, but scripts' exact behavior still unverified by execution | Confirm shipped `.sh` scripts actually reproduce these exact settings |
| Non-RAG baseline ("No Retriever") | Direct LLaVA fine-tune without retrieval | `build_nonrag_dataset.py`, `vqa/*.sh` | Query JSON only | Generated reports, no retrieval | NOT_STARTED | Medium | Not yet fully read |
| Generation metrics | ROUGE-L, BERTScore, F1CheXbert (micro, 5-class), F1RadGraph (partial/RG_ER) | `src/evaluation.py` | Reference + predicted texts | Metric scores | NOT_STARTED | High | BLEU-4 computed in code but not reported in paper — keep as extra, not a comparison target |
| Significance testing | Paper reports "p-value < 0.05" vs. best baseline, method unspecified | Not in official code | Per-example metric scores | p-value | NOT_STARTED | Low (method unknown) | Which test the paper used; our own choice must be documented as such |
| Bootstrap CI | Not reported in paper at all | N/A | Per-example metric scores | Confidence intervals | NOT_STARTED | N/A (`PROPOSED_EXTENSION`) | Method choice is entirely our own |
| Oracle baseline | Precisely defined (Appendix A.3): argmax over corpus of `F1RadGraph+F1CheXbert` instance score, self-excluded for train queries | Not implemented in shipped code as a standalone script | Labeled corpus | Oracle-selected reference per query | NOT_STARTED | High | Reuse this exact definition, don't invent our own "oracle" |
| Adaptive multi-report retrieval (Innovation A) | Not in paper | N/A | Trained retriever + confidence signal | Variable-k retrieval | NOT_STARTED | N/A (`PROPOSED_EXTENSION`) | — |
| Fact-aware reranking (Innovation B) | Not in paper | N/A | Retrieved candidates + factual scores | Reranked candidates | NOT_STARTED | N/A (`PROPOSED_EXTENSION`) | — |
| Multi-report fusion (Innovation C) | Not in paper | N/A | Multiple retrieved reports | Structured evidence context | NOT_STARTED | N/A (`PROPOSED_EXTENSION`) | — |
| Confidence gating (Innovation D) | Not in paper | N/A | Retrieval confidence signal | Retrieve/no-retrieve decision | NOT_STARTED | N/A (`PROPOSED_EXTENSION`) | — |
| Uncertainty/contradiction detection (Innovation E) | Not in paper | N/A | Retrieved facts + generated report | Warnings/confidence summary | NOT_STARTED | N/A (`PROPOSED_EXTENSION`) | — |
| Longitudinal context (Innovation F) | Not in paper (explicitly out of scope, single-study only) | N/A | Prior-study data (if available) | Longitudinal-aware generation | NOT_STARTED | N/A (`PROPOSED_EXTENSION`, optional) | Depends on legal/structural availability of prior studies |

See `docs/paper_analysis.md` §17 for the full Table 1 reported-results
numbers (F1CheXbert/F1RadGraph/ROUGE-L/BERTScore per model, both datasets)
— these are the actual targets our reproduction will be compared against
in Milestone 2.6, not placeholders.
