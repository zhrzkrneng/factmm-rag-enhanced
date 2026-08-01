# Reproduction Matrix

Status legend: `NOT_STARTED` / `PLANNED` / `IN_PROGRESS` / `IMPLEMENTED` /
`TESTED` / `BLOCKED`. `TESTED` requires an actually-observed test run —
in this project's case, both in the author's sandbox and independently
confirmed in the user's own Colab environment — never just "the code
looks right" (per CLAUDE.md's rule against claiming a component works
without execution).

**Milestone 2.1 (data pipeline) is now complete and `TESTED`** — see
the first three rows below. Milestone 2.2 (RadGraph processing)
**annotation-pipeline implementation** (`src/baseline/radgraph/
annotator.py`) has **not** begun — the "RadGraph annotation" and
"CheXbert labeling" rows below remain `NOT_STARTED` for that actual
project code. What *has* happened, and is now complete and tested, is
the **compatibility layer** underneath it: a full environment/
dependency audit (Cell 14, see `docs/colab_execution_log.md`)
confirming both `radgraph==0.0.9` and `f1chexbert==0.0.2` construct and
run correctly on this project's actual Colab environment, after 6
compatibility shims and 2 cache-path fixes for incompatibilities
between radgraph's ~2022-era code and the modern package ecosystem;
those 6 shims have since been moved into version-controlled,
unit-tested code (`src/baseline/radgraph/compat.py`, 29/29 tests
passing) and independently re-verified end-to-end via a second,
self-contained Cell 14 rerun (real `RadGraph()`/`F1CheXbert()`
construction and inference against the actual checkpoints, in a fresh
Colab runtime). This substantially de-risks implementation but is not
itself the annotation-pipeline implementation. One open item surfaced
by the rerun: F1CheXbert produced a different single-class label for
the identical smoke-test input across the two runs — logged in
`docs/risk_register.md` #15, to be resolved before Milestone 2.6, not
blocking Milestone 2.2.

Now that `paper/factmm_rag.pdf` has been read (see `docs/paper_analysis.md`
for full detail), most rows below have a real "Paper description" and
higher confidence than the original code-only pass. Two genuine
paper-vs-code discrepancies were found and are called out explicitly
rather than silently resolved (see the `top_k` and `τ` rows).

| Component | Paper description | Official-code location | Required inputs | Expected outputs | Status | Confidence | Open questions |
|---|---|---|---|---|---|---|---|
| Data parsing | Concatenate finding+impression as report text; select frontal view per study (§3.2 impl. details) | `data/parse.py` | Per-line `.tok` image-path / finding / impression files | `{image, finding, impression}` JSON records | **TESTED** — `ReportRecord` (src/data/schema.py), `JsonReportParser` (src/data/parsing.py); 34/34 tests pass, confirmed in author sandbox + user's Colab | High | Deviation: one `JsonReportParser` class (parameterized by dataset), not separate `MimicCxrParser`/`CheXpertParser` — see docs/progress.md |
| Frontal-view selection | Explicitly stated: "we select the frontal view" | Implicit: `data.py` always uses `image[0]` | Multi-image study records | Single primary image per study | **TESTED** — `ReportRecord.frontal_image_path` (image[0] convention); unit-tested | High (resolved by paper) | Metadata-based (non-convention) frontal selection not yet implemented — still an open TODO for when real MIMIC-CXR metadata is available |
| Patient/study split | Uses MIMIC-CXR's standard processed split (125,417/991/1,624 per Delbrouck et al. 2023) | Not explicit in code; split inherited externally | Dataset with patient/study identifiers | Patient-disjoint train/valid/test manifests | **TESTED** — `IntegrityChecker` + `PatientSplitValidator` (src/data/integrity.py, src/data/splits.py) + `ManifestBuilder` (src/data/manifest.py); unit + end-to-end integration tests pass | Medium | Only tested against synthetic fixtures so far — real MIMIC-CXR/CheXpert data not yet available (see docs/data_requirements.md) |
| RadGraph annotation | RadGraph extracts entities+relations (no version specified) | `data/label.py`, pinned `radgraph==0.0.9` | Report `finding` text | Entities/relations JSON | NOT_STARTED (compatibility layer COMPLETE AND TESTED — `src/baseline/radgraph/compat.py`, 29/29 unit tests, verified end-to-end via Cell 14 rerun; `annotator.py` itself not started) | Medium | Exact model version behind paper's numbers vs. package `0.0.9`; env audit found `transformers` must be `4.57.6` not `4.23.1` (Python 3.12 wheel constraint, see `docs/risk_register.md` #8c) and requires 6 compatibility shims (see #8b, now Mitigated) — moved into version-controlled code |
| CheXbert labeling | 5-class subset used for both mining and evaluation (Cardiomegaly, Edema, Consolidation, Atelectasis, Pleural Effusion) | `data/label.py` | Report `finding` text | 14-class labels, 5-class subset kept | NOT_STARTED (compatibility layer COMPLETE AND TESTED — see above; `annotator.py` itself not started) | High | Checkpoint (`chexbert.pth`, 1.25GB) confirmed publicly downloadable, no auth needed; same cache-path bug as RadGraph found and fixed (`docs/risk_register.md` #8b); F1CheXbert output showed a run-to-run discrepancy on identical input — see `docs/risk_register.md` #15, must be resolved before Milestone 2.6 |
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
