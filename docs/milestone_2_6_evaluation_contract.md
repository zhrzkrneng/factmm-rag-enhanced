# Milestone 2.6 — Evaluation Subsystem: Design Contract

> **DESIGN CONTRACT — NOT EXECUTABLE CODE.** Every class, function, and
> config shown below is a signature-level specification for Cells
> 30–35+, not implemented source. No file under `src/` is created or
> modified by this document. Produced per explicit instruction: audit
> the paper and official repository line-by-line first, distinguish
> official/inferred/unknown behavior throughout, and stop for approval
> before any implementation begins.

## Table of Contents

1. [Status & Scope](#1-status--scope)
2. [Source Audit — Paper](#2-source-audit--paper)
3. [Source Audit — Official Repository](#3-source-audit--official-repository)
4. [Metric Classification](#4-metric-classification)
5. [Architecture Overview](#5-architecture-overview)
6. [Module Layout & Responsibility Boundaries](#6-module-layout--responsibility-boundaries)
7. [Dataclasses & Configs](#7-dataclasses--configs)
8. [Metric Wrappers](#8-metric-wrappers)
9. [Evaluation Pipelines](#9-evaluation-pipelines)
10. [Oracle Baseline Construction](#10-oracle-baseline-construction)
11. [JSON Schemas](#11-json-schemas)
12. [Resume Behavior](#12-resume-behavior)
13. [Deterministic Behavior](#13-deterministic-behavior)
14. [Error Taxonomy](#14-error-taxonomy)
15. [Output Reports](#15-output-reports)
16. [Testing Strategy](#16-testing-strategy)
17. [Colab Execution Plan (Cells 30–35+)](#17-colab-execution-plan-cells-3035)
18. [Known Risks & Paper/Code Discrepancies](#18-known-risks--papercode-discrepancies)
19. [Open Questions (`UNKNOWN`)](#19-open-questions-unknown)

---

## 1. Status & Scope

This is Milestone 2.6's proposed design, produced through a line-by-line
audit of `paper/factmm_rag.pdf` (freshly re-extracted via
`pdftotext -layout`, 843 lines, this session) and every
evaluation-related file in
`reference/original_repository/FactMM-RAG/`: `src/evaluation.py`,
`src/evaluation.sh`, `src/retriever/DPR/evaluate_retriever.py`,
`src/generator/knn_index_to_evaluation_file.py`,
`src/generator/llava_json_to_evaluation_file.py`,
`src/generator/convert_json_or_jsonl.py`,
`src/generator/evaluate_llava.sh`, `src/generator/vqa/eval_llava_vqa.sh`,
`src/generator/knn_ideal.py`,
`data/factual_mining/build_pos_train/gen_similarity.py`,
`data/factual_mining/build_pos_train/gen_similarity.sh`,
`data/factual_mining/build_pos_train/gen_topk_pos.py`,
`data/factual_mining/build_pos_train/gen_topk_oracle_train.sh`,
`data/factual_mining/build_pos_train/merge_topk_pos.sh` — plus the
project's own pre-existing evaluation stubs (`src/evaluation/*.py`,
`src/baseline/evaluation/*.py`, `src/innovation/evaluation/*.py`),
`docs/paper_analysis.md` §15–17, and this project's already-implemented
Milestone 2.2/2.3/2.4/2.5 infrastructure that Milestone 2.6 must reuse
rather than reimplement (`src/baseline/radgraph/{annotator,compat}.py`,
`src/baseline/pair_mining/mining.py`, `src/retrieval/index.py`).

**Baseline only. No implementation yet.** No file under `src/` is
created or modified by this contract. Cell 30 begins only after this
document is approved.

---

## 2. Source Audit — Paper

Section 4 ("Experimental Setup") + Appendix A.3 ("Evaluation Details")
are the paper's evaluation sections. Verbatim/near-verbatim findings,
all `PAPER_EXPLICIT`:

- **Language fluency**: ROUGE-L (longest common subsequence F-measure)
  and BERTScore (semantic similarity).
- **Clinical accuracy**: F1CheXbert — "the F1-score for 5 observations
  (Cardiomegaly, Edema, Consolidation, Atelectasis, Pleural Effusion) by
  comparing the generated report with the reference report's
  classifications." F1RadGraph — "overlap in radiological entities and
  clinical relations between the generated report and the reference
  report."
- **F1-RadGraph (Appendix A.3)**: "we follow previous work
  (MIMIC-CXR-RRS) in employing RGER as F1-Radgraph score computation on
  an instance level... this equates to utilizing
  `reward_level="partial"`."
- **F1-CheXbert (Appendix A.3)**: dataset-level score is "the
  micro-averaged F1-score between 5 selected classes from the CheXbert
  labeler," "only computable over entire datasets." A **separate**
  instance-level formula is defined for pair mining: "the proportion of
  equivalent predicted classes between a reference and predicted text
  sample... computed using `np.sum(ref == hyp) / 5`," taking values in
  `{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}`. **These are two different formulas
  sharing one metric name** — flagged again in §18, this is a real
  footgun independently confirmed against the code (§3).
- **Oracle Retrieval (Appendix A.3)**: "performed via ground-truth
  access to a reference document's generated report... an oracle
  retriever would obtain documents as
  `Oracle(qi) = argmax_{j∈corpus, j≠i} s(qi, dj)`, where `s(q,d)` is the
  **sum of the F1-RadGraph and F1-CheXbert instance-wise scores**." This
  is a **different formula from Equation 1** (§3.1's report-pair-mining
  similarity, F1RadGraph-structural-overlap only, already implemented in
  `src/baseline/pair_mining/mining.py` / `similarity.py`) — Oracle uses
  a **sum of two independent instance-level scores**, not the
  Equation-1 normalized-overlap score. This distinction is confirmed
  independently in the code audit (§3) and is a required new
  computation, not something Milestone 2.2/2.3 already provides.
  Training-time excludes self (`j≠i`); test-time retrieval performs "the
  same operation, without the restriction of `j≠i` as self-retrieval is
  not possible due to the corpus being the training set" — i.e. corpus
  is always train, queries are val/test, so `j≠i` is structurally
  impossible and the exclusion is a no-op at test time, not a separate
  code path.
- **Statistical significance (Table 1 caption)**: "FactMM-RAG
  outperforms the best baseline with p-value < 0.05" — the paper *does*
  perform a significance test, but **names no test** (not paired
  t-test, not bootstrap, not permutation — unspecified) and reports
  **no confidence intervals at all**. Already correctly flagged in
  `docs/paper_analysis.md` §15 and in `src/evaluation/significance.py`'s
  own docstring as a `PROPOSED_EXTENSION`: this project's bootstrap-CI
  approach is *a* reasonable choice, not a reproduction of a specified
  procedure.
- **BLEU**: paper Table 1/Table 2 report only F1CheXbert, F1RadGraph,
  ROUGE-L, BERTScore. **BLEU-4 is never mentioned anywhere in the
  paper** — confirmed by a full-text search of the freshly-extracted
  843-line paper text. It appears only in the official code (§3, §4).
- **Table 1** (reproduction target, `PAPER_EXPLICIT`, already
  transcribed exactly in `docs/paper_analysis.md` §17): 10 rows ×
  2 datasets (MIMIC-CXR, CheXpert zero-shot) × 4 metrics, including a
  `No Retriever` non-RAG baseline and an `Oracle` row (F1CheXbert 0.972 /
  0.951, F1RadGraph 0.523 / 0.384 on MIMIC-CXR / CheXpert respectively —
  far above every real system, confirming Oracle is a ceiling, not a
  competitive baseline).
- **Table 2** (`PAPER_EXPLICIT`, Ablation Study): a separate
  "Multimodal Retrieval" setting (retriever alone, no generator) with
  its own Oracle row (F1CheXbert 0.992/0.999) — **retrieval-only Oracle
  uses a different, higher ceiling than the generation-Oracle row in
  Table 1**, consistent with retrieval-only scoring being closer to a
  true ceiling (no generation-model degradation in between).
- **CheXpert Hidden Test Set (Appendix A.3)**: "We use the 1000 hidden
  test reports from MIMIC-CXR-RRS and download the CheXpert images from
  Stanford AIMI Shared Datasets" — CheXpert zero-shot evaluation reuses
  MIMIC-CXR-RRS's *report text*, pairing it with separately-downloaded
  CheXpert *images*. Not itself an evaluation-metric detail, but
  relevant to how `ref_path`/`pred_path` inputs for the CheXpert
  zero-shot row must be constructed.
- **Limitations (Section 7, `PAPER_EXPLICIT`)**: the paper's own
  self-critique of its evaluation choices — "F1CheXbert reflects
  high-level observational accuracy, while F1RadGraph assesses the
  correctness of radiology entities and clinical relationships. However,
  other radiologically-specific metrics, such as report conciseness and
  clarity, should also be considered... Ideally, we should incorporate
  methods of evaluation directly aligned with human evaluations."
  Directly relevant scope-bounding for this milestone: it does not
  attempt human evaluation or conciseness/clarity metrics, matching the
  paper's own metric set exactly (plus the code-only BLEU-4 extra).

---

## 3. Source Audit — Official Repository

### 3.1 `src/evaluation.py` (generation-quality, `OFFICIAL_REPOSITORY`)

Single, non-resumable, single-pass script. Reads `ref_path`/`pred_path`
JSON files (lists of dicts), pairs `hyps`/`refs` by **raw positional
`zip()`** (no ID-based join), with a silent per-example filter:

```python
for ref, pred in zip(ref_dict, pred_dict):
    hyp = pred['retrieved_finding'][0]
    check = [" ".join(_.split()) for _ in hyp.split(".") if len(_) > 0]
    if len(check) > 0:
        hyps.append(pred['retrieved_finding'][0])
        refs.append(ref['finding'])
assert len(hyps) == len(refs)
```

An example is silently **dropped from both lists** (not scored as 0, not
logged) whenever splitting `hyp` on `.` yields zero non-empty fragments
(i.e. `hyp` is empty/whitespace-only). This is positional pairing with a
silent count-desync risk if `ref_path`/`pred_path` are not already in
lockstep order — the `assert` only catches a *length* mismatch, not a
*content* mismatch (row 5 of `refs` could silently pair with row 5 of a
different query's `hyps` if either file's row order ever drifts). Flagged
as a fragility candidate in §18.

Five metrics computed, **all via direct external-library calls** (see
§4):

```python
f1radgraph = F1RadGraph(reward_level=args.radgraph_level, model_type="radgraph")
score, _, _, _ = f1radgraph(hyps=hyps, refs=refs)

f1chexbert = F1CheXbert(device=args.device)
_, _, _, class_report_5 = f1chexbert(hyps=hyps, refs=refs)
# reported: class_report_5["micro avg"]["f1-score"]

rouge = Rouge()
scores = rouge.get_scores(hyps, refs, avg=True)
# reported: scores['rouge-l']['f']

bleu = evaluate.load("bleu")
results = bleu.compute(predictions=hyps, references=[[r] for r in refs])
# reported: results['precisions'][3]  -- the 4-gram precision component,
# NOT the geometric-mean BLEU score itself

bert_scorer = BERTScorer(
    model_type=args.bert_model, num_layers=5, batch_size=64, nthreads=4,
    all_layers=False, idf=False, device=args.device, lang='en',
    rescale_with_baseline=True, baseline_path=None,
)
_, _, f = bert_scorer.score(cands=hyps, refs=refs)
# reported: f.mean().item()
```

CLI defaults (`src/evaluation.sh`): `REF_PATH=/FactMM-RAG/data/mimic/test.json`,
`PRED_PATH=/FactMM-RAG/data/mimic/test_generated.json`, `DEVICE=cuda`,
`RADGRAPH_LEVEL=partial`, `BERT_MODEL=distilbert-base-uncased`.

### 3.2 `src/retriever/DPR/evaluate_retriever.py` (retrieval-quality, `OFFICIAL_REPOSITORY`)

- `k ∈ {100, 200, 500, 1000}` — a much larger range than typical top-N
  IR evaluation, reflecting the retrieval-for-augmentation (not
  top-1-relevant) setting.
- **Recall@k, NDCG@k**: delegated to `pytrec_eval.RelevanceEvaluator(qrel, {'ndcg_cut', 'recall'})`
  — an external library call, in-process.
- **MRR@k**: a **hand-rolled `compute_mrr` function**, not delegated to
  any library:

  ```python
  def compute_mrr(qids_to_relevant_passageids, qids_to_ranked_candidate_passages, MaxMRRRank=10):
      MRR = 0
      for qid in qids_to_ranked_candidate_passages:
          if qid in qids_to_relevant_passageids:
              target_pid = qids_to_relevant_passageids[qid]       # list, possibly >1 positive
              candidate_pid = qids_to_ranked_candidate_passages[qid]
              for i in range(0, MaxMRRRank):
                  if candidate_pid[i] in target_pid:
                      MRR += 1 / (i + 1)
                      break
      return MRR / len(qids_to_relevant_passageids)
  ```

  Scores the **first** ranked hit against a (possibly multi-element) set
  of relevant IDs — standard MRR against multiple positives, computed
  independently at each of the four `k` cutoffs (not derived from one
  another).
- **Ground truth input**: a pickled `query_positive_matrix` — a list
  (indexed by integer query id) of lists of positive corpus indices.
  Queries with zero positives are **dropped entirely** (not
  zero-scored) from both `query_positive_id` and `ctx_idxs` before
  scoring — `eval_query_cnt` only counts queries that were actually
  scored.
- **Retrieval**: real `faiss.IndexFlatIP` cosine/inner-product search,
  `topN=1000`, `faiss.omp_set_num_threads(16)`.
- **Output**: appended (not overwritten) to a tab-separated `.txt` file
  named `chexbert_{thresh}_radgraph_{thresh}_top1000.txt`, header
  written once if the file doesn't yet exist. CLI defaults
  `--chexbert_threshold=1`, `--radgraph_threshold=0.4` — these are
  purely **descriptive** of which upstream mining run produced the
  `query_positive_matrix` being scored (the script itself does no
  thresholding); they are not consumed as filtering parameters inside
  this script.
- **This project's `PairMiningConfig`** (`src/baseline/pair_mining/mining.py`)
  already has `chex_threshold: float = 1.0` and `radg_threshold: float =
  0.4` as its own defaults — confirmed to match these CLI defaults
  exactly, i.e. the project's existing pair-mining defaults were already
  calibrated against this exact script.

### 3.3 Oracle baseline construction (multi-script pipeline, `OFFICIAL_REPOSITORY`)

Traced end-to-end across 4 scripts, confirming Appendix A.3's formula
(§2) is implemented as a genuinely separate, expensive, offline
pipeline — **not** a simple function call inside `evaluation.py`:

1. **`data/factual_mining/build_pos_train/gen_similarity.py`**
   (`gen_similarity.sh`, a SLURM array job, 64 shards): computes the
   **full O(N×N) pairwise** `chexbert_similarity`/`radgraph_similarity`
   matrix for the entire corpus (`n=125417` for the training corpus),
   one shard per array task, `--cpus-per-task=1`, `--mem=16G`,
   `--time=03:00:00` *per shard* — reuses the same `utils.chexbert_similarity`/
   `utils.radgraph_similarity` helper functions as the project's own
   already-implemented pair-mining similarity code
   (`src/baseline/pair_mining/similarity.py`), confirming Oracle
   construction is a different **use** of the same underlying scoring
   primitives, not a different metric implementation.
2. **`gen_topk_pos.py`** (`gen_topk_oracle_train.sh`): loads the
   `chex_i.npy`/`radg_i.npy` shards, computes `tensor = chexbert + radgraph`
   (elementwise sum — this **is** Appendix A.3's `s(qi,dj)` formula,
   confirmed in code), masks by `pre_mask_chex`/`pre_mask_radg`
   thresholds (both `0.0` for the Oracle's own exhaustive-search
   invocation — no filtering at all, unlike real report-pair mining's
   `1.0`/`0.4`), then `np.argpartition` for `top_k=30` per query. Output:
   one pickle per shard, `{"positive_list": [[idx, ...], ...], ...}`.
3. **`knn_ideal.py`**: merges the 64 shard pickles into one file shaped
   identically to a real KNN-retriever's output —
   `[{"key": i, "knn_index": [...]}]` — the **same generic shape**
   consumed by the real-retriever path.
4. **`knn_index_to_evaluation_file.py`**: consumes *either* a real
   embedding-KNN `knn_index` pickle *or* this Oracle-ranked one — same
   script, same code path, no branching:

   ```python
   output = [
       {"retrieved_finding": [train_data[knn_data_i["knn_index"][0]]["finding"]]}
       for knn_data_i in knn_data
   ]
   ```

   Takes **index `[0]`** (top-1) of whichever ranking was supplied,
   verbatim, as the "prediction" — **zero generation** for the Oracle
   path; the "prediction" is literally the top-ranked corpus report's
   ground-truth text. This confirms the Oracle mechanism precisely: it
   is not a real-embedding KNN lookup at all, despite the shared script
   name — it is an argmax over a **ground-truth text-similarity** score
   matrix, formatted into the same generic index shape purely so the
   downstream conversion script can be reused unchanged.

`Oracle-LLaVA` (Appendix A.3, Table 2's `Oracle` retrieval-augmented
generation row) is a distinct third configuration: LLaVA fine-tuned
under identical conditions but using Oracle-retrieved documents (not
real-retriever documents) as the RAG context, both at train and test
time — i.e. the Oracle mechanism feeds a full RAG fine-tuning run
identical in every other respect to the real pipeline, not just a
one-shot eval-time substitution.

### 3.4 Shared conversion pipeline (`convert_json_or_jsonl.py`, `evaluate_llava.sh`, `vqa/eval_llava_vqa.sh`)

Both the RAG-conditioned and VQA-only (no-RAG) ablation paths run the
**identical** 3-step CLI: `convert_json_or_jsonl.py` (generic bidirectional
`.json`/`.jsonl` converter) → `llava_json_to_evaluation_file.py` (wraps
real LLaVA `text` output into `[{"retrieved_finding": [<text>]}]`) →
`evaluation.py` — differing only in input/output file paths. Confirms
`evaluation.py` is the single, shared scoring entry point for every
generation-quality row in both Table 1 and Table 2's RAG-augmented
section.

---

## 4. Metric Classification

Per explicit requirement #5 — every metric labeled by where its
computation actually happens:

| Metric | Classification | Detail |
|---|---|---|
| F1RadGraph | **Delegated to external library** (in-process) | `radgraph.F1RadGraph`, package `radgraph==0.0.9` (already pinned + shimmed in `src/baseline/radgraph/compat.py`, Milestone 2.2) |
| F1CheXbert (dataset-level) | **Delegated to external library** (in-process) | `f1chexbert.F1CheXbert`, package `f1chexbert==0.0.2` (already pinned + shimmed, Milestone 2.2) |
| F1CheXbert (instance-level, pair-mining/Oracle/significance) | **Own formula, implemented in official repo's `utils.py`** — this project's equivalent already exists in `src/baseline/pair_mining/similarity.py` | `np.sum(ref==hyp)/5`, not a library call at all |
| ROUGE-L | **Delegated to external library** (in-process) | `rouge.Rouge` (the `rouge` PyPI package — distinct from Google's `rouge-score` / HF `evaluate`'s `"rouge"`, must not be substituted) |
| BLEU-4 | **Delegated to external library** (in-process) | `evaluate.load("bleu")` (HuggingFace `evaluate`); `OFFICIAL_REPOSITORY`-only, not `PAPER_EXPLICIT` (§2) |
| BERTScore | **Delegated to external library** (in-process) | `bert_score.BERTScorer` |
| Recall@k, NDCG@k | **Delegated to external library** (in-process) | `pytrec_eval.RelevanceEvaluator` |
| MRR@k | **Implemented inside the repository** — own hand-rolled function, not delegated | official `evaluate_retriever.py::compute_mrr` |
| FAISS retrieval | **Delegated to external library** (in-process) | `faiss.IndexFlatIP` — this project already wraps this identically in `src/retrieval/index.py` |
| Oracle score `s(q,d)` (F1RadGraph+F1CheXbert instance sum) | **Implemented inside the repository** — own formula (`tensor = chexbert + radgraph`) | `gen_topk_pos.py`; the instance-level F1RadGraph/F1CheXbert values it sums are themselves computed by `utils.py`'s own formulas, not by the library-level `F1RadGraph`/`F1CheXbert` classes used for §3.1's dataset-level scoring |
| Oracle pairwise score-matrix construction (`gen_similarity.py`) | **Executed through an external script pipeline** | A SLURM array job (`gen_similarity.sh`, 64 shards) — not a single in-process function call at all; genuinely batch/offline, hours of compute |

No metric in this subsystem is executed via a subprocess/shell-out from
within a single Python evaluation call (unlike, say, invoking a CLI
tool via `subprocess.run`) — the "external script" classification
applies only to the **Oracle pairwise-matrix precomputation step**,
which the official repo runs as a genuinely separate offline SLURM
pipeline, not something `evaluation.py` or `evaluate_retriever.py`
themselves shell out to.

---

## 5. Architecture Overview

```
                    ┌──────────────────────────┐
                    │  GenerationMetricsConfig  │
                    │  RetrievalMetricsConfig   │
                    │  SignificanceConfig        │
                    │  EvaluationRunConfig       │
                    └────────────┬──────────────┘
                                 │
        ┌────────────────────────┼─────────────────────────┐
        ▼                        ▼                          ▼
┌───────────────────┐  ┌─────────────────────┐   ┌──────────────────────┐
│ generation_metrics │  │ retrieval_metrics    │   │ significance.py       │
│ .py (src/evaluation)│  │ .py (src/evaluation) │   │ (src/evaluation)       │
│ F1RadGraph/CheXbert/ │  │ MRR@k / Recall@k /   │   │ PROPOSED_EXTENSION:   │
│ ROUGE-L/BLEU-4/     │  │ NDCG@k, wraps        │   │ paired bootstrap CI   │
│ BERTScore wrappers  │  │ FaissFlatIPIndex     │   │ over per-example      │
└──────────┬──────────┘  └──────────┬───────────┘   │ scores                │
           │                        │                └───────────┬───────────┘
           └───────────┬────────────┘                            │
                        ▼                                        │
              ┌───────────────────┐                              │
              │    report.py       │◄─────────────────────────────┘
              │ (src/evaluation)   │
              │ EvaluationReport    │
              │ JSON + CSV output   │
              └─────────┬───────────┘
                        │
        ┌────────────────┼─────────────────────┐
        ▼                                       ▼
┌────────────────────────┐          ┌─────────────────────────────┐
│ src/baseline/evaluation/ │          │ src/innovation/evaluation/    │
│ comparison.py             │          │ innovation_metrics.py          │
│ vs. paper Table 1          │          │ Phase-5 ablation stats          │
│ (docs/paper_analysis.md   │          │ (adaptive-k, gating rate,       │
│ §17), incl. Oracle row     │          │  fusion agreement)               │
└────────────────────────┘          └─────────────────────────────┘

Oracle construction (separate, expensive, offline-style pipeline):
  PairMiner-style instance scorer → OracleConfig-driven top-1 argmax
  → same JSONL "prediction" schema as real generation output (§11)
  → scored by the exact same generation_metrics.py pipeline above
```

`generation_metrics.py`/`retrieval_metrics.py` never construct an
`EvaluationReport` themselves — `report.py` is the sole aggregation
point, mirroring `_generator_result_to_jsonl_row`'s "one function
composes the outcome" discipline from Milestone 2.5.

---

## 6. Module Layout & Responsibility Boundaries

Matches the pre-existing Phase-1 stub file locations exactly — **no
reorganization**, unlike Milestone 2.5's deliberate `src/generation/` →
`src/baseline/generation/` move:

```
src/evaluation/
    __init__.py             # unchanged shared package docstring
    generation_metrics.py   # F1RadGraph/F1CheXbert/ROUGE-L/BLEU-4/BERTScore
                             #   wrappers + dataclasses; malformed-pair
                             #   handling; pure functions over (hyps, refs)
    retrieval_metrics.py    # MRR@k/Recall@k/NDCG@k wrappers + dataclasses;
                             #   consumes FaissFlatIPIndex + positives map
    significance.py         # PROPOSED_EXTENSION: paired bootstrap CI
    report.py                # EvaluationReport aggregation, JSON/CSV writers

src/baseline/evaluation/
    __init__.py
    comparison.py            # comparison table vs. docs/paper_analysis.md
                              #   §17's Table 1 numbers, incl. Oracle row
    oracle.py                # NEW (not a pre-existing stub -- Oracle
                              #   construction has no home in the Phase-1
                              #   scaffold; added here since it is
                              #   baseline-specific per Appendix A.3, not a
                              #   shared metric)

src/innovation/evaluation/
    __init__.py
    innovation_metrics.py    # unchanged scope (Phase 5, out of scope here)
```

**Responsibility boundaries** (strict, mirroring Milestone 2.5's
discipline): `generation_metrics.py`/`retrieval_metrics.py` never
import `radgraph`/`f1chexbert`/`bert_score`/`pytrec_eval`/`evaluate`
at module scope — only inside private default-factory functions,
identical to `RadGraphAnnotator`'s and `HFGeneratorAdapter`'s
established pattern, so unit tests never trigger a real heavy import.
`generation_metrics.py` reuses `src.baseline.radgraph.compat.patch_all`
for its F1RadGraph/F1CheXbert factories rather than re-implementing
compatibility shims — the exact same `radgraph==0.0.9`/`f1chexbert==0.0.2`
packages Milestone 2.2 already pinned and shimmed are the ones
`evaluation.py` itself imports (§3.1); there is no reason for a second,
divergent shim layer. `report.py` contains no metric-computation logic
of its own — it only aggregates already-computed results.
`oracle.py` depends on `src.baseline.pair_mining.similarity`'s
instance-level scoring functions (reused, not reimplemented) and
produces output in the exact schema `generation_metrics.py` consumes
(§11) — it is a *producer* of `pred_path`-shaped data, not a metric
computer itself.

---

## 7. Dataclasses & Configs

All `frozen=True`, `__post_init__`-validated, versioned via
`config_version`, matching every prior milestone's dataclass
convention.

### 7.1 `GenerationMetricsConfig`

```python
@dataclass(frozen=True)
class GenerationMetricsConfig:
    config_version: str = "1.0"
    radgraph_reward_level: str = "partial"          # OFFICIAL_REPOSITORY (evaluation.sh default)
    radgraph_model_type: str = "radgraph"            # matches F1RadGraph(model_type="radgraph")
    chexbert_device: str = "cpu"                      # DEVIATION: official default is "cuda";
                                                       #   this project never silently assumes GPU
                                                       #   presence (see HFGeneratorAdapter precedent)
    bert_score_model_type: str = "distilbert-base-uncased"  # OFFICIAL_REPOSITORY default
    bert_score_num_layers: int = 5
    bert_score_batch_size: int = 64
    bert_score_rescale_with_baseline: bool = True
    bleu_enabled: bool = True                         # OFFICIAL_REPOSITORY extra, NOT paper-reported
                                                       #   (§2) -- always labeled as such in output,
                                                       #   never silently folded into "paper metrics"
    pairing_mode: Literal["by_query_key", "positional_official_reproduction"] = "by_query_key"
    malformed_hypothesis_filter_enabled: bool = False  # project default OFF: a dropped example is
                                                       #   an error (EvaluationError), never silent;
                                                       #   True reproduces official evaluation.py's
                                                       #   silent per-example filter exactly (§3.1, §18)
```

`__post_init__`: non-empty `config_version`/`radgraph_reward_level`/
`radgraph_model_type`/`chexbert_device`/`bert_score_model_type`;
`bert_score_num_layers`/`bert_score_batch_size` positive ints.

### 7.2 `RetrievalMetricsConfig`

```python
@dataclass(frozen=True)
class RetrievalMetricsConfig:
    config_version: str = "1.0"
    k_values: Tuple[int, ...] = (100, 200, 500, 1000)   # OFFICIAL_REPOSITORY exact set
    chexbert_threshold: float = 1.0    # descriptive provenance only (§3.2) -- matches
    radgraph_threshold: float = 0.4    #   PairMiningConfig's own defaults exactly, confirmed (§3.2)
```

`__post_init__`: `k_values` non-empty, all positive ints, strictly
increasing (no duplicates); `chexbert_threshold`/`radgraph_threshold`
non-negative floats.

### 7.3 `OracleConfig`

```python
@dataclass(frozen=True)
class OracleConfig:
    config_version: str = "1.0"
    corpus_scope: str = "train_only"   # matches RAGDatasetBuilderConfig's own field (§2, PAPER_EXPLICIT)
    exclude_self: bool = True          # True for train-time Oracle construction (j != i, PAPER_EXPLICIT
                                        #   Appendix A.3); False for val/test-time (self-retrieval is
                                        #   structurally impossible per Appendix A.3's own wording, §2)
    top_k_candidates_considered: int = 30  # OFFICIAL_REPOSITORY (gen_topk_oracle_train.sh's own --top_k);
                                        #   only rank-0 of these is ever used as the final prediction (§3.3)
                                        #   -- kept for parity/debuggability, not because rank>0 is consumed
```

`__post_init__`: non-empty `config_version`/`corpus_scope`;
`top_k_candidates_considered` a positive int.

### 7.4 `SignificanceConfig`

```python
@dataclass(frozen=True)
class SignificanceConfig:
    config_version: str = "1.0"
    method: Literal["paired_bootstrap"] = "paired_bootstrap"  # PROPOSED_EXTENSION (§2) -- the paper
                                                                #   names no test; this is our own choice
    num_bootstrap_samples: int = 10000
    confidence_level: float = 0.95
    seed: int = 42
```

`__post_init__`: `num_bootstrap_samples` a positive int;
`confidence_level` in `(0, 1)`; `seed` an int.

### 7.5 `EvaluationRunConfig`

```python
@dataclass(frozen=True)
class EvaluationRunConfig:
    config_version: str = "1.0"
    run_id: str                                    # caller-supplied, e.g. UUID or timestamp-derived
    system_name: str                                # e.g. "FactMM-RAG", "Med-MARVEL", "No Retriever", "Oracle"
    dataset_name: Literal["mimic-cxr", "chexpert"]
    split: Literal["train", "valid", "test"] = "test"
    seed: int = 42
```

`__post_init__`: non-empty `config_version`/`run_id`/`system_name`.

### 7.6 Result dataclasses

```python
@dataclass(frozen=True)
class GenerationExampleScore:
    query_key: QueryKey
    f1_radgraph: float
    f1_chexbert_instance: float   # np.sum(ref==hyp)/5 formula (§2, §4) -- NOT the dataset micro-F1
    rouge_l: float
    bert_score: float
    # BLEU-4 deliberately excluded: the official evaluate.load("bleu") API has no per-example
    # decomposition (corpus-level n-gram precision only) -- UNKNOWN/not-applicable per-example,
    # documented in §18/§19, not silently approximated

@dataclass(frozen=True)
class GenerationMetricsResult:
    config_version: str
    num_examples_scored: int
    num_examples_dropped: int          # malformed-hypothesis filter / pairing mismatches
    f1_radgraph: float                  # corpus-level (library's own aggregate)
    f1_chexbert: float                  # dataset-level micro-avg F1 over 5 observations
    rouge_l: float
    bleu4: Optional[float]              # None when bleu_enabled=False
    bert_score: float
    per_example: Optional[Tuple[GenerationExampleScore, ...]]  # None unless requested (needed for significance.py)

@dataclass(frozen=True)
class RetrievalMetricsResult:
    config_version: str
    num_queries_evaluated: int
    num_queries_dropped_no_positives: int
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    mrr_at_k: Dict[int, float]

@dataclass(frozen=True)
class SignificanceResult:
    metric_name: str
    system_a_name: str
    system_b_name: str
    system_a_mean: float
    system_b_mean: float
    mean_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    num_bootstrap_samples: int
    seed: int

@dataclass(frozen=True)
class EvaluationReport:
    run_config: EvaluationRunConfig
    generation_metrics: Optional[GenerationMetricsResult]
    retrieval_metrics: Optional[RetrievalMetricsResult]
    significance: Optional[Tuple[SignificanceResult, ...]]
    package_versions: Dict[str, str]
    hardware: str
    git_commit: Optional[str]
    generated_timestamp_utc: str
```

---

## 8. Metric Wrappers

Every wrapper follows the established injected-factory discipline
(`HFGeneratorAdapter`/`RadGraphAnnotator` precedent): no real library
constructed by default inside a unit test, real construction reserved
for a dedicated dry-run cell.

```python
# src/evaluation/generation_metrics.py

def compute_f1_radgraph(
    hyps: Sequence[str], refs: Sequence[str], config: GenerationMetricsConfig,
    *, f1radgraph_factory: Optional[Callable[[], object]] = None,
) -> Tuple[float, Tuple[float, ...]]:
    """Returns (corpus_score, per_example_scores). Wraps radgraph.F1RadGraph
    exactly as evaluation.py does (§3.1); reuses compat.patch_all() via the
    default factory."""

def compute_f1_chexbert_dataset(
    hyps: Sequence[str], refs: Sequence[str], config: GenerationMetricsConfig,
    *, f1chexbert_factory: Optional[Callable[[], object]] = None,
) -> float:
    """Dataset-level micro-avg F1 over the 5-class subset (§2, §3.1). No
    per-example decomposition -- the library itself only returns an
    aggregate classification report."""

def compute_f1_chexbert_instance(hyp: str, ref: str, *, chexbert_labeler) -> float:
    """np.sum(ref_labels == hyp_labels) / 5 (§2 Appendix A.3, §4) -- reuses
    src.baseline.pair_mining.similarity's already-implemented instance-level
    CheXbert scoring, not a fresh reimplementation."""

def compute_rouge_l(hyps: Sequence[str], refs: Sequence[str],
                     *, rouge_factory: Optional[Callable[[], object]] = None,
                     ) -> Tuple[float, Tuple[float, ...]]: ...

def compute_bleu4(hyps: Sequence[str], refs: Sequence[str],
                   *, bleu_factory: Optional[Callable[[], object]] = None,
                   ) -> float:
    """Corpus-level only (results['precisions'][3], §3.1) -- no per-example
    return value; the HF evaluate "bleu" metric provides no such
    decomposition."""

def compute_bert_score(hyps: Sequence[str], refs: Sequence[str], config: GenerationMetricsConfig,
                        *, bert_scorer_factory: Optional[Callable[[], object]] = None,
                        ) -> Tuple[float, Tuple[float, ...]]: ...

def evaluate_generation(
    ref_rows: Sequence[dict], pred_rows: Sequence[dict], config: GenerationMetricsConfig,
    *, factories: Optional[GenerationMetricsFactories] = None,
) -> GenerationMetricsResult:
    """Orchestrator: pairs ref/pred (by query_key or official positional
    reproduction per config.pairing_mode), applies the malformed-hypothesis
    filter iff config.malformed_hypothesis_filter_enabled, calls all 5
    metric wrappers, returns one GenerationMetricsResult."""
```

```python
# src/evaluation/retrieval_metrics.py

def compute_mrr_at_k(
    positives: Dict[QueryKey, Tuple[QueryKey, ...]],
    ranked: Dict[QueryKey, Tuple[QueryKey, ...]],
    k: int,
) -> float:
    """Own reimplementation of the official compute_mrr (§3.2), ported from
    integer pids/pickle format to this project's QueryKey typing. Same
    semantics: first-hit-in-top-k against a (possibly multi-element)
    positive set, independently computed per k (not derived from a larger-k
    result)."""

def compute_recall_ndcg_at_k(
    positives: Dict[QueryKey, Tuple[QueryKey, ...]],
    ranked: Dict[QueryKey, Tuple[QueryKey, ...]],
    k_values: Tuple[int, ...],
    *, pytrec_eval_factory: Optional[Callable[[dict], object]] = None,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Delegates to pytrec_eval.RelevanceEvaluator (§3.2, §4), in-process.
    Returns (recall_at_k, ndcg_at_k)."""

def evaluate_retrieval(
    index: FaissFlatIPIndex, query_keys: Sequence[QueryKey], query_embeddings: np.ndarray,
    positives: Dict[QueryKey, Tuple[QueryKey, ...]], config: RetrievalMetricsConfig,
) -> RetrievalMetricsResult:
    """Orchestrator: index.search(query_embeddings, max(config.k_values))
    (reusing FaissFlatIPIndex.search()'s existing clipping-not-erroring
    behavior for corpora smaller than max(k_values), §5 of Milestone 2.5's
    own precedent), drops queries with zero positives (matching official
    behavior exactly, §3.2), calls compute_mrr_at_k / compute_recall_ndcg_at_k
    per k, returns one RetrievalMetricsResult."""
```

```python
# src/evaluation/significance.py  (PROPOSED_EXTENSION)

def paired_bootstrap_significance(
    system_a_scores: Sequence[float], system_b_scores: Sequence[float],
    metric_name: str, system_a_name: str, system_b_name: str,
    config: SignificanceConfig,
) -> SignificanceResult:
    """Deterministic given config.seed via a LOCAL numpy.random.Generator
    instance (numpy.random.default_rng(config.seed)) -- NEVER the global
    np.random/torch global RNG state. This is a direct, disclosed lesson
    from this session's own test_retrieval_trainer.py flakiness bug (fixed
    by seeding before construction, not after) -- the same seed-BEFORE-use
    discipline applies here: the Generator must be constructed fresh inside
    this function from config.seed, never reused/mutated across calls,
    so two calls with the same seed always produce byte-identical CIs
    regardless of call order or prior global RNG state."""
```

---

## 9. Evaluation Pipelines

### 9.1 Generation-quality pipeline

```
RAGDatasetBuilder / LLaVAGenerator output (Milestone 2.5 JSONL)
        │  _generator_result_to_jsonl_row rows (query_key, generated_report_text, ...)
        ▼
  convert to pred_rows: [{"query_key": [...], "retrieved_finding": [text]}]
        │  (this project's own converter, replacing convert_json_or_jsonl.py +
        │   llava_json_to_evaluation_file.py -- same target schema, §11)
        ▼
  ref_rows: [{"query_key": [...], "finding": text}]  (ground truth from data/schema.py's ReportRecord)
        ▼
  evaluate_generation(ref_rows, pred_rows, GenerationMetricsConfig)
        ▼
  GenerationMetricsResult  ──►  report.py  ──►  EvaluationReport
```

### 9.2 Retrieval-quality pipeline

```
Trained retriever (Milestone 2.4) + FaissFlatIPIndex (already built)
        │
        ▼
  Query embeddings (test/valid split) ──► index.search(top_k=max(k_values))
        │
  Positives: src.baseline.pair_mining.mining's QueryMiningResult.positive_keys
             (this project's OWN substitute for the official pickled
             query_positive_matrix -- same semantics, JSONL serialization
             instead of pickle, §18)
        ▼
  evaluate_retrieval(index, query_keys, query_embeddings, positives, RetrievalMetricsConfig)
        ▼
  RetrievalMetricsResult  ──►  report.py  ──►  EvaluationReport
```

### 9.3 Ablation-setting reuse

Per §3.4's confirmed finding, the **same** `evaluate_generation`
pipeline serves every generation-quality row across both Table 1 (RAG)
and Table 2's retrieval-only / backbone-variation / VQA-only-ablation
rows — only which JSONL feeds `pred_rows` changes, never the pipeline
itself. This mirrors the official repo's own single shared
`evaluation.py` entry point exactly (§3.1, §3.4).

---

## 10. Oracle Baseline Construction

`src/baseline/evaluation/oracle.py` (§6, new module — no pre-existing
stub to reconcile with):

```python
def compute_oracle_score(chex_instance_score: float, radg_instance_score: float) -> float:
    """s(qi, dj) = F1CheXbert_instance + F1RadGraph_instance, verbatim
    Appendix A.3 / gen_topk_pos.py's `tensor = chexbert + radgraph` (§2, §3.3).
    NOT Equation 1's normalized-overlap formula -- a distinct computation,
    confirmed independently from both the paper text and the code."""

def build_oracle_predictions(
    query_records: Sequence[ReportRecord], corpus_records: Sequence[ReportRecord],
    config: OracleConfig,
    *, chexbert_instance_scorer: Callable[[str, str], float],
    radgraph_instance_scorer: Callable[[str, str], float],
    output_path: Path, meta_path: Path,
    continue_on_error: bool = False, errors_path: Optional[Path] = None,
) -> OracleBuildSummary:
    """For each query, computes s(qi, dj) against every corpus candidate
    (excluding self per config.exclude_self), takes argmax, emits one JSONL
    row in the exact pred_rows schema (§9.1, §11) with the winning corpus
    candidate's finding text as the "prediction" -- zero generation,
    matching knn_index_to_evaluation_file.py's mechanism exactly (§3.3).
    Resume/atomic-write/continue_on_error identical to PairMiner's
    established pattern (§12) -- the 5th implementation of this pattern in
    this project."""
```

This is a genuinely expensive O(`len(query_records)` × `len(corpus_records)`)
computation — see §18 for the realistic Colab-scale caveat.

---

## 11. JSON Schemas

### 11.1 Generation-eval input (`ref_rows` / `pred_rows`)

Deliberately **not** positional lists like the official `ref.json`/
`pred.json` (§3.1) — this project pairs by `query_key` explicitly by
default (§7.1's `pairing_mode`), so both files carry an explicit key:

```json
// ref_rows (one row per ground-truth report)
{"query_key": ["mimic-cxr", "p10000032", "s50414267"], "finding": "..."}

// pred_rows (one row per generated/retrieved report -- same schema for
// real LLaVA output, VQA-only ablation output, AND Oracle output, §9.3, §10)
{"query_key": ["mimic-cxr", "p10000032", "s50414267"], "retrieved_finding": ["..."]}
```

`pairing_mode="positional_official_reproduction"` accepts the official
schema instead (bare `{"finding": ...}` / `{"retrieved_finding": [...]}}`
lists, paired by raw index) for controlled A/B reproduction only.

### 11.2 `EvaluationReport` JSON (written by `report.py`)

```json
{
  "run_config": {
    "config_version": "1.0", "run_id": "...", "system_name": "FactMM-RAG",
    "dataset_name": "mimic-cxr", "split": "test", "seed": 42
  },
  "generation_metrics": {
    "config_version": "1.0", "num_examples_scored": 1624, "num_examples_dropped": 0,
    "f1_radgraph": 0.257, "f1_chexbert": 0.602, "rouge_l": 0.307, "bleu4": null,
    "bert_score": 0.561, "per_example": null
  },
  "retrieval_metrics": null,
  "significance": null,
  "package_versions": {"radgraph": "0.0.9", "f1chexbert": "0.0.2", "rouge": "...", "evaluate": "...", "bert_score": "...", "pytrec_eval": "..."},
  "hardware": "cpu",
  "git_commit": "...",
  "generated_timestamp_utc": "2026-08-05T00:00:00Z"
}
```

### 11.3 Comparison table row (`comparison.py`)

```json
{
  "dataset": "mimic-cxr", "system_name": "FactMM-RAG",
  "reproduced": {"f1_chexbert": 0.602, "f1_radgraph": 0.257, "rouge_l": 0.307, "bert_score": 0.561},
  "paper_reported": {"f1_chexbert": 0.602, "f1_radgraph": 0.257, "rouge_l": 0.307, "bert_score": 0.561},
  "absolute_delta": {"f1_chexbert": 0.0, "f1_radgraph": 0.0, "rouge_l": 0.0, "bert_score": 0.0},
  "paper_table_source": "docs/paper_analysis.md §17 (Table 1)"
}
```

`paper_reported` values are the fixed constants transcribed in
`docs/paper_analysis.md` §17 (`PAPER_EXPLICIT`) — `comparison.py` never
re-derives them from the PDF at runtime, only references the already-
audited, already-committed transcription.

---

## 12. Resume Behavior

**Not uniform across the subsystem** — a deliberate distinction, not an
oversight:

- **`evaluate_generation`/`evaluate_retrieval` themselves are NOT
  resumable.** This matches official behavior exactly (§3.1, §3.2 are
  both single-pass, in-process scripts with no checkpointing) — MIMIC-CXR
  test/CheXpert zero-shot are small (1,624 / 1,000 examples), scoring is
  comparatively cheap even with real model loads, and introducing
  resume machinery for a one-shot scoring pass would be complexity
  without a matching official precedent or a demonstrated need.
- **`build_oracle_predictions` (§10) IS resumable**, using the
  established pattern exactly (`_load_completed_*_keys`,
  `_write_meta_atomic`, `continue_on_error`/`errors_path`, config-
  compatibility-gated resume via `_check_meta_compatibility`'s pattern)
  — the 5th implementation of this same pattern in this project
  (`RadGraphAnnotator`, `PairMiner`, `RAGDatasetBuilder`,
  `LLaVAGenerator`, now `oracle.py`), because Oracle construction is
  genuinely expensive (§10, §18) and interruption-prone.

---

## 13. Deterministic Behavior

- ROUGE-L, BLEU-4, F1RadGraph, F1CheXbert (both formulas), Recall@k,
  NDCG@k, MRR@k: deterministic given fixed input text — no sampling
  involved anywhere in these computations.
- BERTScore: deterministic given a fixed model checkpoint and fixed
  inputs (no dropout/sampling at inference), but exact floating-point
  results can differ marginally across CPU/GPU or batch-size boundaries
  — a real, documented risk (§18), not something this contract can
  eliminate.
- `paired_bootstrap_significance`: deterministic **only** via a local
  `numpy.random.Generator(seed)`, never the global `np.random` or
  `torch` RNG state (§8) — this is a direct, disclosed carry-over of
  the exact lesson learned fixing `test_synthetic_end_to_end_smoke_lifecycle`'s
  real flakiness bug this session (seeding-order hazard: constructing
  something that consumes global RNG state before seeding makes
  "identical seed ⇒ identical result" false). `significance.py`'s test
  suite must include an explicit repeated-run determinism check (≥20
  repetitions, matching the verification discipline used to confirm
  that earlier fix), not just a single passing run.
- `build_oracle_predictions`: deterministic given fixed
  `query_records`/`corpus_records`/scorer functions — `argmax` over a
  real-valued score is well-defined only up to tie-breaking; ties must
  be broken by a fixed, documented rule (lowest corpus index, matching
  `np.argpartition`'s own implicit behavior in the official code as
  closely as is verifiable) rather than left to whatever order a `max()`
  call happens to encounter first.

---

## 14. Error Taxonomy

One new exception class, `EvaluationError`, added to
`src/common/exceptions.py` — matching the project's established
one-class-per-milestone-stage convention exactly (`AnnotationError`,
`PairMiningError`, `RetrievalDatasetError`, `RetrievalIndexError`,
`RetrieverTrainingError`, `RAGDatasetError`), not split further per
sub-module:

| Condition | Raised as |
|---|---|
| `ref_rows`/`pred_rows` count mismatch (`pairing_mode="by_query_key"`: key-set mismatch beyond what the malformed filter accounts for) | `EvaluationError` |
| Malformed JSON schema (missing `query_key`/`finding`/`retrieved_finding`) | `EvaluationError` |
| External library raises during a metric call (`radgraph`/`f1chexbert`/`rouge`/`evaluate`/`bert_score`/`pytrec_eval`) | `EvaluationError`, chained (`raise EvaluationError(...) from exc`), never swallowed |
| `chexbert_device="cuda"` requested but no GPU present | `EvaluationError` — same classified-error discipline as `HFGeneratorAdapter`'s `_classify_generation_error` (§ Milestone 2.5), reusing that classifier's `device_unavailable`-equivalent category rather than inventing a new one |
| Positives file missing/empty for `evaluate_retrieval` | `EvaluationError` |
| `k` in `k_values` exceeds corpus size | **Not an error** — clipped via `FaissFlatIPIndex.search()`'s own existing clipping behavior (Milestone 2.4 precedent), consistent with the rest of this project's search-wrapping code |
| Duplicate `query_key` within `ref_rows` or `pred_rows` | `EvaluationError` |
| `build_oracle_predictions` resume-metadata config mismatch | `EvaluationError` (reuses `_check_meta_compatibility`'s pattern exactly) |
| `significance.py` called with mismatched-length `system_a_scores`/`system_b_scores` (not truly paired) | `EvaluationError` |

---

## 15. Output Reports

`report.py`:

```python
def build_evaluation_report(
    run_config: EvaluationRunConfig,
    generation_metrics: Optional[GenerationMetricsResult] = None,
    retrieval_metrics: Optional[RetrievalMetricsResult] = None,
    significance: Optional[Tuple[SignificanceResult, ...]] = None,
) -> EvaluationReport: ...

def write_report_json(report: EvaluationReport, path: Path) -> None: ...

def write_reports_csv(reports: Sequence[EvaluationReport], path: Path) -> None:
    """One row per EvaluationReport, flattened -- the multi-run comparison
    format (across systems/datasets/splits) used to reconstruct a Table-1-
    shaped view."""
```

`comparison.py`:

```python
def build_comparison_table(
    reports: Sequence[EvaluationReport], paper_table1: Mapping[Tuple[str, str], Mapping[str, float]],
) -> Tuple[ComparisonRow, ...]:
    """paper_table1 keyed by (dataset_name, system_name) -> the fixed
    constants transcribed in docs/paper_analysis.md §17 (§11.3). Includes
    the Oracle row when an Oracle-system EvaluationReport is present,
    labeled distinctly (never averaged into "real system" comparisons,
    since Oracle is a ceiling, not a competitive baseline, §2)."""
```

---

## 16. Testing Strategy

- Every metric wrapper (§8) unit-tested with **injected fake
  factories** returning fixed, hand-computed scores — no real
  `radgraph`/`f1chexbert`/`rouge`/`evaluate`/`bert_score`/`pytrec_eval`
  construction in any unit test, mirroring `HFGeneratorAdapter`'s
  established discipline exactly (§6).
- `evaluate_generation`'s pairing logic tested for both `pairing_mode`
  values independently, including the official malformed-hypothesis
  filter's exact reproduction (`hyp.split(".")` empty-fragment
  behavior) when `malformed_hypothesis_filter_enabled=True`, and its
  absence (raising `EvaluationError` instead) when `False`.
- `compute_mrr_at_k` unit-tested against hand-computed small examples at
  each of the four official `k` values independently — including a
  multi-positive-per-query case (the official function's `target_pid`
  is a list, not a single ID, §3.2) and a query with zero positives
  (dropped, not zero-scored, §3.2).
- `compute_recall_ndcg_at_k` unit-tested against `pytrec_eval` with a
  small fixed qrel/run — real `pytrec_eval` import (a lightweight pure
  scoring library, not a heavyweight ML dependency, so this one library
  is exercised directly in unit tests rather than faked, matching the
  precedent already set by testing `faiss.IndexFlatIP` directly in
  Milestone 2.4's own test suite).
- `compute_oracle_score` unit-tested as a pure two-argument sum;
  `build_oracle_predictions` tested end-to-end against a small synthetic
  corpus (≤10 records) with hand-computed expected argmax winners,
  including a deliberate tie to exercise the documented tie-breaking
  rule (§13).
- `paired_bootstrap_significance` tested for: determinism (≥20 repeated
  calls, same seed, byte-identical `SignificanceResult` — directly
  modeled on this session's own `test_retrieval_trainer.py` flakiness-fix
  verification protocol, §13); a known-identical-systems case (mean
  difference ≈ 0, CI straddles 0); a known-clearly-different-systems
  case (CI excludes 0, `p_value` small).
- `report.py`/`comparison.py` tested as pure functions: fixed
  `EvaluationReport` inputs → exact expected JSON/CSV/comparison-row
  output, including the Oracle-row-labeled-distinctly behavior (§15).
- Resume behavior (`build_oracle_predictions` only, §12): re-running
  skips completed keys, never duplicates output rows, config-mismatch
  resume rejected.
- Static/import-time check: `generation_metrics.py`/`retrieval_metrics.py`
  never import their respective heavy libraries at module scope
  (AST-based check, mirroring `test_hf_generator_adapter.py`'s
  established pattern, §6).
- One real-library dry run per external dependency — reserved for a
  dedicated Colab cell (§17), never run as part of the standard unit
  test suite.

---

## 17. Colab Execution Plan (Cells 30–35+)

- **Cell 30** — `src/evaluation/generation_metrics.py` (all 5 metric
  wrappers with injected fakes, `evaluate_generation` orchestrator,
  dataclasses) + unit tests. New `radgraph`/`f1chexbert` real imports
  reuse Milestone 2.2's already-installed environment (Cell 14/15) — no
  reinstall.
- **Cell 31** — `src/evaluation/retrieval_metrics.py` (`compute_mrr_at_k`,
  `compute_recall_ndcg_at_k` against real `pytrec_eval`,
  `evaluate_retrieval` orchestrator) + unit tests. Requires adding
  `pytrec_eval` to `requirements.txt` (not yet present — flagged §18,
  not done by this contract).
- **Cell 32** — `src/evaluation/significance.py` (`paired_bootstrap_significance`)
  + unit tests, including the ≥20-repetition determinism check (§16).
- **Cell 33** — `src/baseline/evaluation/oracle.py` (`compute_oracle_score`,
  `build_oracle_predictions`) + unit tests against a small synthetic
  corpus only — **no real 125k-scale run** (§10, §18).
- **Cell 34** — `src/evaluation/report.py` +
  `src/baseline/evaluation/comparison.py` (aggregation, JSON/CSV
  writers, comparison table vs. `docs/paper_analysis.md` §17) + unit
  tests.
- **Cell 35 — Real Dependency Dry Run** (mirrors Milestone 2.4G/2.5's
  Cell-22/29 pattern exactly): real `F1RadGraph`/`F1CheXbert` calls
  reusing Cell 14/15's already-verified environment, real `Rouge`,
  real `evaluate.load("bleu")`, real `BERTScorer` construction (a
  `distilbert-base-uncased`-based checkpoint download — small relative
  to LLaVA's 7B, but still a real network+memory event, gated behind
  the same preflight-reachability check established in Milestone
  2.4G/2.5), real `pytrec_eval` scoring on a small synthetic qrel/run —
  all against synthetic data only, never real patient reports at this
  stage. Ends in `CELL 35: PASS`/`FAIL`.
- Full-scale Oracle construction (§10) and the actual Table-1
  reproduction run (real MIMIC-CXR/CheXpert data, real trained
  retriever + generator checkpoints from Milestones 2.4/2.5) are
  **out of scope for this contract's Colab cells** — they depend on
  real checkpoints this project has not yet produced (full fine-tuning
  is Colab-infeasible per the Milestone 2.5 contract's own §13) and are
  deferred to whatever milestone actually runs the end-to-end
  reproduction, not invented here.

---

## 18. Known Risks & Paper/Code Discrepancies

- **F1CheXbert name overload** (§2, §3.1, §4): the same metric name
  denotes two different formulas — dataset-level micro-F1 (final
  reporting) vs. instance-level `np.sum(ref==hyp)/5` (pair mining,
  Oracle, per-example significance testing). Every dataclass/function
  in this contract names these `f1_chexbert` vs. `f1_chexbert_instance`
  explicitly and never lets one silently stand in for the other.
- **Positional pairing + silent malformed-hypothesis filter** in
  official `evaluation.py` (§3.1) is a genuine fragility: a length-only
  `assert` cannot catch a same-length-but-shifted pairing. This
  project's default (`pairing_mode="by_query_key"`,
  `malformed_hypothesis_filter_enabled=False`) is a real, disclosed
  deviation from official behavior — official reproduction remains
  available opt-in, matching the exact `strict_*` vs.
  `reproduce_official_bug` pattern already established in Milestone 2.5
  §4.
- **BLEU-4 is `OFFICIAL_REPOSITORY`-only**, never paper-reported (§2,
  §4) — must never be silently included in any "matches paper Table 1"
  comparison claim; `comparison.py` (§15) only compares the 4
  paper-reported metrics against `docs/paper_analysis.md` §17.
- **Oracle construction is computationally massive at real scale**
  (§3.3, §10): the official pipeline parallelizes an
  O(125,417 × 125,417) pairwise score matrix across a 64-shard SLURM
  array, each shard budgeted 3 hours. Even the smaller test-scale case
  (1,624 test queries × 125,417 train corpus ≈ 204M pairs) is far
  beyond a single interactive Colab session. This contract's Colab plan
  (§17, Cell 33) deliberately scopes real execution to a small synthetic
  corpus only — a full-scale Oracle run is a genuine open risk requiring
  either long-running background compute or a subsampled corpus, not
  solved by this contract.
- **Statistical significance methodology is a `PROPOSED_EXTENSION`**
  (§2, §7.4): the paper's own "p-value < 0.05" claim names no test and
  reports no CIs — `significance.py`'s paired-bootstrap approach is
  this project's own methodological choice, to be labeled as such in
  every output, never presented as reproducing a specified procedure.
- **BERTScore/CheXbert real-model dry run carries the same
  network/memory risk class** already encountered twice in this project
  (Milestone 2.4G, Milestone 2.5 Cell 29's real OOM near-miss) — Cell
  35 (§17) must reuse the established preflight-reachability +
  measured-VRAM/RAM-safety-threshold pattern, not skip it because these
  checkpoints are individually smaller than LLaVA's 7B.
- **New external dependencies not yet in `requirements.txt`**:
  `pytrec_eval`, `rouge`, `evaluate` (HF), `bert_score` are all required
  by this milestone and are currently absent from `requirements.txt`
  (only `radgraph`/`f1chexbert` are already installed, per Milestone
  2.2's Cell 14). Adding them is an implementation-time action, not
  performed by this contract.
- **`rouge` (PyPI) vs. `rouge-score` naming collision**: the official
  code imports `from rouge import Rouge` — the `rouge` PyPI package,
  distinct from Google's `rouge-score` package or HuggingFace
  `evaluate`'s built-in `"rouge"` metric (which wraps `rouge-score` and
  computes ROUGE differently). Installing the wrong package would
  silently produce numerically different ROUGE-L scores — flagged
  explicitly so implementation does not substitute one for the other.
- **CheXpert zero-shot input construction** (§2): reuses MIMIC-CXR-RRS's
  1,000 hidden-test *report* text paired with separately-downloaded
  CheXpert *images* — an asymmetric-source detail with its own
  data-availability dependency (Stanford AIMI Shared Datasets), outside
  this evaluation-subsystem contract's own scope but a precondition for
  ever running the CheXpert zero-shot row for real.
- **BERTScore CPU/GPU floating-point drift** (§13): exact bit-for-bit
  reproducibility across hardware is not guaranteed even at a fixed
  seed/checkpoint — a real, unresolved caveat for any strict
  reproduction claim, not unique to this project's implementation.

---

## 19. Open Questions (`UNKNOWN`)

- Exact tie-breaking behavior of the official `np.argpartition`-based
  Oracle top-k selection when multiple corpus candidates share the
  identical maximal `s(qi,dj)` score — the official code's
  `argpartition` has an implementation-defined tie order; this
  contract's `build_oracle_predictions` documents an explicit
  lowest-index rule (§13) as a reasonable, disclosed choice, not a
  verified match to NumPy's internal tie behavior.
- Whether the official `evaluation.py`'s positional-pairing fragility
  (§18) has ever actually caused a silent scoring error in the paper's
  own reported numbers — unknowable without access to the paper
  authors' exact `ref_path`/`pred_path` file generation order.
- Per-example BLEU-4 decomposition: whether any reasonable
  per-example proxy exists for feeding BLEU into
  `paired_bootstrap_significance` — currently treated as genuinely
  not applicable (§7.6), not attempted.
- Real GPU memory profile for the Cell 35 dry run (§17) — unmeasured
  until an actual run is observed, consistent with every prior
  milestone's own disclosed-estimate-only stance on unmeasured
  hardware claims.
- Whether `docs/reproduction_matrix.md` needs a new evaluation-subsystem
  row once this milestone is implemented — not checked as part of this
  contract, flagged for the implementation turn.
