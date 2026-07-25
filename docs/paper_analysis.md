# Paper Analysis: FactMM-RAG

## Paper Access Status — READ THIS FIRST

**`paper/factmm_rag.pdf` does not exist in this repository yet** (only a
`.gitkeep` placeholder is present). Per CLAUDE.md, no implementation detail
may be fabricated, so this document does **not** claim to summarize the full
paper text.

Two independent attempts were made to obtain the paper directly:

1. Cloning the official GitHub repository (`cxcscmu/FactMM-RAG`) —
   **succeeded**. Its `README.md`, scripts, and configuration values are used
   throughout this document and are labeled `OFFICIAL_REPOSITORY`.
2. Fetching the arXiv abstract/HTML page (`arxiv.org/abs/2407.15268`) —
   **blocked by the sandbox's outbound network policy** (the proxy rejects
   `CONNECT` to `arxiv.org` with a 403; `github.com` is allowed). This is an
   environment/policy restriction, not a code problem, and it was not
   bypassed.

**Action required from the user:** upload `paper/factmm_rag.pdf` (or paste
its text) so the sections below can be completed and re-labeled
`PAPER_EXPLICIT` where appropriate. Until then, every claim in this document
is sourced from the official repository's code/README, or is explicitly
marked `UNKNOWN`. Nothing here should be treated as a verified paper claim.

Bibliographic facts below **are** confirmed, because they come from the
official repo's own `README.md` citation block (`OFFICIAL_REPOSITORY`):

| Field | Value | Source |
|---|---|---|
| Title | Fact-Aware Multimodal Retrieval Augmentation for Accurate Medical Radiology Report Generation | OFFICIAL_REPOSITORY |
| Venue | NAACL 2025 | OFFICIAL_REPOSITORY |
| arXiv ID | 2407.15268 | OFFICIAL_REPOSITORY |
| Authors | Liwen Sun, James Zhao, Megan Han, Chenyan Xiong | OFFICIAL_REPOSITORY |
| Official repo | https://github.com/cxcscmu/FactMM-RAG | OFFICIAL_REPOSITORY |
| License (official code) | MIT | OFFICIAL_REPOSITORY |

---

## 1. Research Problem

- **OFFICIAL_REPOSITORY**: The repo README states the goal as "a fact-aware
  multimodal retrieval-augmented pipeline for generating accurate radiology
  reports" — i.e., using retrieval of similar prior reports to condition
  report generation from chest X-ray images, with retrieval driven by
  *factual* (not just semantic) similarity.
- **UNKNOWN**: The paper's full motivating narrative, related-work framing,
  and precise problem statement wording — pending PDF.

## 2. Claimed Contributions

- **UNKNOWN** (pending PDF). The official repo's README schedule/checklist
  implies at least three deliverable components were released: data
  preprocessing, factual report-pair mining, and retriever training code,
  with generator training code explicitly marked as a separate, later
  release. This suggests the retriever and its factual-pair-mining strategy
  are a central contribution, but this is `REASONABLE_INFERENCE`, not a
  quoted claim.

## 3. Complete Pipeline (as implemented in the official repository)

`OFFICIAL_REPOSITORY` — reconstructed by reading the code, in execution order:

1. **Parsing** (`data/parse.py`): combine per-line image path(s), findings,
   and impressions text files into a single JSON list of
   `{"image": [...], "finding": ..., "impression": ...}` records. Only the
   **first** listed image path is used downstream as the primary image.
2. **Annotation** (`data/label.py`): run `RadGraph` (entities/relations) and
   `F1CheXbert` (14-class labels) over each report's `finding` text; store a
   5-class label subset (`Cardiomegaly, Edema, Consolidation, Atelectasis,
   Pleural Effusion`) plus RadGraph entities per report.
3. **Factual similarity scoring** (`data/factual_mining/build_pos_*/gen_similarity.py`):
   for every query report against every corpus report, compute:
   - **CheXbert similarity**: fraction of the 5 labels that match exactly.
   - **RadGraph similarity**: an "exact entity token if relation exists"
     F1-style reward comparing entity+label (and relation-existence) sets
     between two reports (`utils.py:exact_entity_token_if_rel_exists_reward`).
   Computed as an all-pairs matrix, chunked for parallel (SLURM array) jobs.
4. **Top-k positive mining** (`gen_topk_pos.py` / `merge_topk_pos.py`):
   combine CheXbert + RadGraph similarity, apply a **pre-mask** requiring
   `chexbert_sim >= chex_thresh` **and** `radgraph_sim >= radg_thresh`, then
   take the top-`k` remaining candidates as positive references for each
   query. Self-retrieval is explicitly detected and optionally dropped
   (`--ignore_self`); rows whose self-similarity looks degenerate can be
   flagged as "bad samples" and skipped (`--skip_bad_sample`).
   Production values used in the shipped `.sh` scripts:
   `chex_thresh=1.0`, `radg_thresh=0.4`, `top_k=3`.
5. **Retriever training** (`src/retriever/DPR`, then `src/retriever/ANCE`):
   a dual/multi-modal encoder (see §7) trained with in-batch-negative
   contrastive loss against the mined positive pairs; a second ANCE-style
   stage adds mined **hard negatives**.
6. **Embedding export + indexing** (`gen_embeddings.py`, FAISS `IndexFlatIP`
   / `IndexIVFFlat`): encode all images/reports, build a FAISS index over
   corpus (typically training-set) report embeddings.
7. **Retrieval / KNN construction** (`src/generator/knn.py`,
   `src/retriever/DPR/retrieval.py`): for each query image embedding,
   retrieve nearest corpus report embeddings.
8. **RAG dataset construction** (`src/generator/build_rag_dataset.py`):
   pick the highest-ranked retrieved candidate that is **not** from the same
   study or the same patient (explicit patient/study-leakage filter) and,
   optionally, not "too short"; embed it into a fixed prompt template (see
   §9); build LLaVA-format conversational (train) or inference (test) JSON.
9. **Generation** (LLaVA fine-tuning/inference, `install_llava.sh`,
   `train_llava.sh`, `inference_llava.sh`): a vision-language model
   generates a report from the image + retrieved-report-conditioned prompt.
10. **Evaluation** (`src/evaluation.py`): F1RadGraph, F1CheXbert (5-class
    micro-F1), ROUGE-L, BLEU-4, BERTScore over generated vs. reference text.

## 4. Datasets

- **OFFICIAL_REPOSITORY**: MIMIC-CXR (via the vilmedic ACL-2023 preprocessed
  split, per README link) and CheXpert (Stanford AIMI). `data/mimic/` holds
  `train.json` / `valid.json` / `test.json`; `data/chexpert/` holds
  `test.json` only in the shipped repo (only a 1-record schema example is
  present in both — real data is not distributed with the code).
- **OFFICIAL_REPOSITORY**: the shipped SLURM script
  `gen_topk_pos.sh` hard-codes `--n 125417`, implying the MIMIC-CXR training
  split used contained **125,417** report records.
- **UNKNOWN**: exact CheXpert split sizes, exact MIMIC-CXR train/valid/test
  counts as reported in the paper's dataset table, and any additional
  filtering criteria beyond what's visible in code.

## 5. Preprocessing

- **OFFICIAL_REPOSITORY**: reports are split into `finding` and
  `impression` fields upstream of this repo (already split before
  `parse.py` runs); `parse.py` only merges parallel line-based `.tok` files.
- **OFFICIAL_REPOSITORY**: only the **first** path in a study's `image`
  list is used as the "primary" image in `parse.py`'s output entry
  construction is at the raw-file level (multiple paths kept in the JSON
  `image` array), but the retriever's `MedDataset` always indexes
  `example['image'][0]` — i.e., a single (frontal, by MIMIC-CXR convention)
  view is used for encoding. **REASONABLE_INFERENCE**: this implements the
  "frontal view" selection mentioned in our project rules, but the paper's
  own justification/verification of frontal-view selection is `UNKNOWN`.

## 6. RadGraph Annotation

- **OFFICIAL_REPOSITORY**: uses the `radgraph` PyPI package (pinned
  `radgraph==0.0.9` in `requirements.txt`), invoked as `RadGraph()([report])`,
  returning per-report `entities` (with `tokens`, `label`, `relations`).
- **UNKNOWN**: which underlying RadGraph model checkpoint/version
  (`radgraph` vs `radgraph-xl`, inference vs. reward variant) is used;
  the package's default is treated as authoritative here but this should be
  re-confirmed once the paper is available, since RadGraph has multiple
  released variants.

## 7. Factual Pair Mining

Covered in §3 steps 3–4. Configurable knobs identified in code
(`OFFICIAL_REPOSITORY`):
- `chexbert_threshold` (default `1` in mining scripts, i.e. exact match of
  all 5 labels)
- `radgraph_threshold` (default `0.4`)
- `top_k` positives per query (`3` in shipped scripts)
- `ignore_self`, `skip_bad_sample` toggles

## 8. Retriever Architecture

`OFFICIAL_REPOSITORY` (`src/retriever/DPR/multi_model.py`):
- Vision tower: `CLIPVisionModel` (default `openai/clip-vit-base-patch32`),
  patch tokens (excluding CLS) linearly projected into the text model's
  hidden size.
- Text/query tower: `T5ForConditionalGeneration`
  (default `OpenMatch/t5-ance` — the **MARVEL** checkpoint referenced in the
  README, "Checkpoint: MARVEL", is loaded as `--pretrained_model_path`).
- Fusion: image patch embeddings are spliced into the T5 encoder's input
  embedding sequence in place of image placeholder tokens
  (`<im_start><im_patch>*N<im_end>`), i.e. a MARVEL/LLaVA-style
  patch-token-splicing scheme, not a separate late-fusion pooling.
- A single `logit_scale` parameter (initialized from a `CLIPModel` instance)
  scales the similarity matrix, CLIP-style.
- Query representation = image only (`forward(images, text_inputs=None)`).
  Candidate representation = image + prompt-wrapped report text
  (`forward(images, text_inputs)`), i.e. retrieval is *image-to-report*
  where the candidate side is also conditioned on its own image.
- **UNKNOWN**: whether the paper describes an alternative/ablated
  text-only or image-only retriever variant beyond what's in code.

## 9. Generator Architecture

`OFFICIAL_REPOSITORY`:
- LLaVA (vision-language instruction-tuned LLM), per `install_llava.sh`
  (`haotian-liu/LLaVA`, pinned to a specific commit) — trained/fine-tuned
  via `train_llava.sh`, inference via `inference_llava.sh`.
- Exact prompt template (`build_rag_dataset.py`):
  `Here is a report of a related patient: "{retrieved_doc}"\nGenerate a
  radiology report from this image:<image>` (conversational/training form),
  with the reference answer being the query's own `finding` or `impression`
  text (`output_data_mode`).
- **UNKNOWN**: the generator training code was explicitly marked
  **unreleased** in the README schedule at clone time ("Release the
  generator training code" is unchecked) even though `train_llava.sh` /
  `inference_llava.sh` scripts exist — treat generator reproducibility as
  higher-risk pending manual verification (see risk register).

## 10. Training Objectives

`OFFICIAL_REPOSITORY` (`src/retriever/DPR/train.py`):
- In-batch contrastive loss: cosine similarity (L2-normalized embeddings)
  scaled by `logit_scale.exp()`, cross-entropy against the diagonal
  (matching query/positive pairs), i.e. a standard InfoNCE/CLIP-style
  symmetric-free (single-direction: query→candidate) softmax loss.
- Optimizer: AdamW, `betas=(0.9, 0.98)`, `eps=1e-6`; weight decay `0.2` on
  ≥2-D non-bias/norm/logit_scale params, `0.0` elsewhere (CLIP-style
  parameter grouping).
- LR `5e-6`, cosine schedule with warmup (`warmup_steps=0.1`, interpreted as
  a *fraction* of total steps), `num_train_epochs=15`, batch size `32`,
  early stopping on dev accuracy (`patience=5` eval rounds).
- ANCE stage (`src/retriever/ANCE/train.py`) — **not yet read in full**;
  based on `train.sh` it re-initializes from the DPR checkpoint and adds
  `--train_neg_path` / `--valid_neg_path` hard-negative files.

## 11. Negative Sampling

`OFFICIAL_REPOSITORY`:
- **Stage 1 (DPR)**: in-batch negatives only (other candidates in the same
  training batch).
- **Stage 2 (ANCE)**: adds mined hard negatives
  (`gen_hard_negatives.py`) — for each query, retrieve top-`N=100` corpus
  candidates by current DPR embeddings via FAISS, keep only candidates
  whose CheXbert similarity `< 1.0` **and** RadGraph similarity `< 0.4`
  (i.e., confirmed factually-dissimilar despite embedding closeness), sort
  ascending by combined score, and keep the `num_top_neg=2` lowest-scoring
  (most clearly negative) of those as hard negatives per query.

## 12. Retrieval Strategy (Inference)

`OFFICIAL_REPOSITORY`:
- FAISS `IndexFlatIP` (exact) or `IndexIVFFlat` (approximate, `nlist`
  configurable) over corpus embeddings; top-`k` search (`knn.py` default
  `k=20`, `results_k=5` kept in output; `retrieval.py` hard-codes `3`).
- Final report selection (`build_rag_dataset.py`) walks the ranked
  candidates and picks the **first one that is not the same study and not
  the same patient** as the query (and, optionally, not "too short"); if
  none qualifies, falls back to the raw top-1 and logs an "anomaly".
- Baseline generation therefore uses exactly **one** retrieved report per
  query — multi-report retrieval is not present in the official baseline.

## 13. Generation Strategy

Covered in §9. Single retrieved report concatenated into a fixed natural
language prompt prefix, then standard LLaVA image+text generation.

## 14. Baselines

- **UNKNOWN** (pending PDF) for the paper's own reported baseline set
  (e.g. non-RAG LLaVA, other RAG variants). The repo does provide a
  **non-RAG VQA** LLaVA training path (`build_nonrag_dataset.py`,
  `vqa/train_llava_vqa.sh`) as an apples-to-apples no-retrieval comparison
  — `OFFICIAL_REPOSITORY`.

## 15. Evaluation Metrics

`OFFICIAL_REPOSITORY` (`src/evaluation.py`):
- F1RadGraph (`reward_level="partial"` by default)
- F1CheXbert, micro-avg F1 over the same 5-class subset used in mining
- ROUGE-L (F-measure, via `Rouge` package)
- BLEU-4 (`evaluate` library `bleu` metric, `precisions[3]`)
- BERTScore (F1, `distilbert-base-uncased` by default, layer 5,
  `rescale_with_baseline=True`)
- Retrieval-side: MRR@{100,200,500,1000}, Recall@{100,200,500,1000},
  NDCG@{100,200,500,1000} (`evaluate_retriever.py`, via `pytrec_eval`).
- **Not present in official code**: bootstrap confidence intervals or
  significance testing — these are requested project additions
  (`PROPOSED_EXTENSION`, see Milestone 2.6), not part of the original
  evaluation.

## 16. Ablation Studies

**UNKNOWN** — pending PDF. No ablation-running scripts were found in the
official repository beyond the DPR-vs-ANCE two-stage retriever comparison
and the RAG-vs-non-RAG-VQA generator comparison already noted.

## 17. Reported Results

**UNKNOWN** — pending PDF. No results tables exist in the official
repository (only a results-file *format* is defined in
`evaluate_retriever.py`'s tab-separated output writer).

## 18. Implementation Details (confirmed from code)

- Retriever-stage pinned environment: `torch==1.13.1`,
  `transformers==4.23.1`, `faiss-cpu==1.10.0`, `radgraph==0.0.9`,
  `f1chexbert==0.0.2`, `wandb==0.16.4` (`requirements.txt`).
- Generator-stage (LLaVA) environment is **separate and incompatible**:
  `install_llava.sh` upgrades to `transformers==4.36.2`, `peft==0.10.0`,
  `accelerate==0.21.0`, `tokenizers==0.15.1` in its own conda env
  (`OFFICIAL_REPOSITORY`) — this project must plan for two isolated
  environments, not one.
- Training used SLURM (`#SBATCH --gres=gpu:A6000:1`, 60G mem, 1-day time
  limit for retriever training) — `OFFICIAL_REPOSITORY`, informs
  `docs/compute_requirements.md`.

## 19. Missing or Ambiguous Details

- Full paper text (motivation, related work, exact contribution list,
  ablations, headline numbers) — **UNKNOWN**, pending PDF upload.
- RadGraph model variant/version pinning beyond the PyPI package version.
- Exact MIMIC-CXR/CheXpert split sizes and any exclusion criteria applied
  before the `125,417`-row training corpus was reached.
- Whether "frontal view selection" is an explicit, separately-validated
  preprocessing step in the paper, or simply a byproduct of always taking
  `image[0]`.
- Generator training code was marked unreleased in the README checklist at
  clone time, despite `.sh` scripts existing in `src/generator/` — treat as
  needing manual verification, not as a confirmed-working reference
  implementation.

## 20. Limitations & Reproducibility Risks

See `docs/risk_register.md` for the structured register. Headline risks:
dataset access is credentialed (MIMIC-CXR, CheXpert) and cannot be
auto-downloaded; the MARVEL checkpoint requires a HuggingFace download;
RadGraph inference requires PhysioNet-credentialed model weights in some
distributions; the generator stage depends on a separate, pinned-commit
LLaVA fork; two mutually-incompatible Python environments are required
(retriever vs. generator).

---

## Source Labels Used

- `PAPER_EXPLICIT` — directly stated in the paper text (none yet, pending PDF)
- `OFFICIAL_REPOSITORY` — verified by reading the cloned official code/README
- `REASONABLE_INFERENCE` — inferred from code/README with explicit reasoning
- `PROPOSED_EXTENSION` — not in paper or official code; our own addition
- `UNKNOWN` — genuinely unknown pending more source material
