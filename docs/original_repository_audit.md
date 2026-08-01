# Official Repository Audit — cxcscmu/FactMM-RAG

## Provenance

- Source: `https://github.com/cxcscmu/FactMM-RAG` (public, MIT licensed).
- Cloned into: `reference/original_repository/FactMM-RAG/` on 2026-07-25.
- Clone method: `git clone` over HTTPS (network reachable to `github.com`
  through the sandbox's proxy).
- The nested `.git/` directory from the clone was **removed** after cloning
  so the vendored snapshot is stored as plain tracked files rather than a
  git submodule/gitlink (avoids a broken nested-repo reference when this
  project's own repository is committed). No file **content** was modified.
- Default branch cloned: `main` (repo also has a `knn_rag` branch and at
  least one open PR, `#2`, not fetched — only `main`'s tip was cloned).
- This directory is treated as **read-only reference material**. Nothing
  under `reference/original_repository/` should ever be edited by this
  project; adaptations belong in `src/baseline/`.

## Top-Level Structure

```
FactMM-RAG/
├── LICENSE                     (MIT)
├── README.md                   (setup + usage instructions, citation)
├── requirements.txt            (retriever-stage pinned deps)
├── install_llava.sh            (separate generator-stage env setup)
├── assets/overview.png         (pipeline diagram, not parsed — image)
├── data/
│   ├── parse.py                 # merge image/finding/impression -> JSON
│   ├── label.py                 # RadGraph + CheXbert annotation
│   ├── mimic/{train,valid,test}.json   # schema-only placeholders (1 record each)
│   ├── chexpert/test.json              # schema-only placeholder
│   └── factual_mining/
│       ├── build_pos_train/     # similarity + top-k mining, training split
│       └── build_pos_valid/     # same, for validation split (query=valid, corpus=train)
└── src/
    ├── evaluation.py             # F1RadGraph/F1CheXbert/ROUGE-L/BLEU-4/BERTScore
    ├── evaluation.sh
    ├── retriever/
    │   ├── checkpoint/model.best.pt        # empty placeholder (0 bytes)
    │   ├── DPR/  (data.py, multi_model.py, train.py, train.sh,
    │   │          gen_embeddings.py, gen_hard_negatives.py,
    │   │          retrieval.py, evaluate_retriever.py,
    │   │          embedding/embedding [empty], output/dpr.best.pt [empty])
    │   └── ANCE/ (data.py, multi_model.py, train.py, train.sh,
    │              gen_embeddings.py,
    │              embedding/embedding [empty], output/ance.best.pt [empty],
    │              retrieval/retrieved_reports.json [empty])
    └── generator/
        ├── build_rag_dataset.py         # RAG prompt/dataset construction
        ├── build_nonrag_dataset.py      # non-RAG VQA dataset construction
        ├── knn.py / knn_ideal.py        # FAISS KNN index construction
        ├── knn_index_to_evaluation_file.py
        ├── llava_json_to_evaluation_file.py
        ├── convert_json_or_jsonl.py
        ├── train_llava.sh / inference_llava.sh / evaluate_llava.sh
        └── vqa/{train,inference,eval}_llava_vqa.sh
```

66 files total, 712 KB on disk (`du -sh`) — no dataset, checkpoint, or
embedding file exceeds a few bytes; every `.pt`, `embedding`, and
`retrieved_reports.json` file present is an **empty placeholder** (0 bytes),
confirming the repository intentionally ships no real weights or data.

## File-by-File Notes (files actually read in full)

| File | Role | Key facts extracted |
|---|---|---|
| `data/parse.py` | Build `{"image","finding","impression"}` JSON from parallel line files | Only `image_paths[0]` becomes the primary image path in each record |
| `data/label.py` | RadGraph + CheXbert annotation | Uses `radgraph` and `f1chexbert` PyPI packages; keeps a 5-class CheXbert label subset + RadGraph entities |
| `data/factual_mining/build_pos_train/utils.py` | Similarity functions | `chexbert_similarity` = fraction of 5 labels matching; `radgraph_similarity` = entity+relation F1-style reward |
| `.../gen_similarity.py` | All-pairs similarity matrix, chunked | SLURM array job, 64 chunks in shipped `.sh` |
| `.../gen_topk_pos.py` | Threshold + top-k positive selection | Pre-mask by `chex>=thresh` AND `radg>=thresh`; self-retrieval and "bad sample" (diagonal RadGraph ≠ 1) detection |
| `.../merge_topk_pos.py` | Merge chunked pickle outputs | — |
| `.../gen_topk_pos.sh` | Production hyperparameters | `chex_thresh=1.0`, `radg_thresh=0.4`, `top_k=3`, `n=125417` (MIMIC train corpus size) |
| `src/retriever/DPR/multi_model.py` | Retriever architecture | CLIP ViT-B/32 vision tower + T5 (`OpenMatch/t5-ance`, i.e. MARVEL) text tower, patch-token splicing fusion, single `logit_scale` |
| `src/retriever/DPR/data.py` | Retriever dataset | Query = image only; candidate = image + prompt-wrapped report text; in-batch negatives via batch structure |
| `src/retriever/DPR/train.py` | Retriever training loop | In-batch InfoNCE/CLIP loss, AdamW (β=0.9/0.98, eps=1e-6), lr 5e-6, wd 0.2 (non-bias), cosine warmup schedule, bs 32, 15 epochs, early stop patience 5 |
| `src/retriever/DPR/gen_hard_negatives.py` | ANCE hard-negative mining | FAISS top-100 by embedding, keep candidates with chex<1.0 AND radg<0.4, take 2 lowest-scoring as hard negatives |
| `src/retriever/DPR/evaluate_retriever.py` | Retrieval metrics | MRR/Recall/NDCG @ {100,200,500,1000} via `pytrec_eval` |
| `src/retriever/DPR/retrieval.py` | Simple top-3 retrieval script | FAISS `IndexFlatIP`, hard-coded `k=3` |
| `src/generator/knn.py` | General KNN index builder | FAISS `IndexFlatIP` or `IndexIVFFlat`, configurable `k`/`nlist`/`nprobe` |
| `src/generator/build_rag_dataset.py` | RAG prompt construction | **Explicit same-study and same-patient exclusion** when picking the retrieved report; fixed prompt template; falls back to raw top-1 with a logged "anomaly" if all candidates are filtered out |
| `src/evaluation.py` | Final generation metrics | F1RadGraph (partial), F1CheXbert (micro, 5-class), ROUGE-L, BLEU-4, BERTScore (distilbert-base-uncased) |
| `requirements.txt` | Retriever-stage pinned env | torch 1.13.1, transformers 4.23.1, faiss-cpu 1.10.0, radgraph 0.0.9, f1chexbert 0.0.2 |
| `install_llava.sh` | Generator-stage env | Separate conda env; clones `haotian-liu/LLaVA` at a pinned commit; upgrades transformers to 4.36.2 — **incompatible** with the retriever-stage pins |
| `README.md` | Usage + citation | Full command sequence for every stage; NAACL 2025 citation block; checklist shows generator training code marked **not yet released** at clone time |

Files present but **not yet read in full** (lower priority — mirrors of the
DPR versions, or one-off conversion utilities): `src/retriever/ANCE/data.py`,
`src/retriever/ANCE/train.py`, `src/retriever/ANCE/gen_embeddings.py`,
`src/retriever/DPR/gen_embeddings.py`, `src/retriever/DPR/utils.py`,
`data/factual_mining/build_pos_valid/*`,
`src/generator/knn_ideal.py`, `src/generator/build_nonrag_dataset.py`,
`src/generator/convert_json_or_jsonl.py`,
`src/generator/knn_index_to_evaluation_file.py`,
`src/generator/llava_json_to_evaluation_file.py`,
`src/generator/vqa/*.sh`. These should be read before implementing the
corresponding baseline milestone (retriever ANCE stage / generator stage),
not before — consistent with "work milestone by milestone."

## What Is *Not* in the Official Repository

- No real dataset files (MIMIC-CXR, CheXpert) — must be obtained separately
  under each dataset's own license/DUA.
- No model weights (MARVEL checkpoint, DPR/ANCE best checkpoints, LLaVA
  weights) — all `.pt`/`embedding` files are empty placeholders.
- No results tables, logs, or reported metric values.
- No unit tests.
- No configuration-file system (all hyperparameters are CLI args with
  defaults, some overridden only inside `.sh` scripts) — this project's
  requirement to "use configuration files instead of hard-coded values" is
  a deliberate improvement over the official code, not a reproduction of it.
- No `ablation` or `experiments` directory.

## Git-Tracking Note

Two gitignore-interaction issues were caught and fixed before committing:

1. This project's top-level `.gitignore` originally had an unanchored
   `data/` rule (meant to exclude a root-level real-dataset directory) that
   also matched `reference/original_repository/FactMM-RAG/data/` anywhere in
   the tree, silently dropping real source code (`parse.py`, `label.py`, all
   `factual_mining/` scripts) from version control. Fixed by anchoring the
   rule to the repo root (`/data/`, `/datasets/`).
2. The vendored repo's **own** `.gitignore` (kept as-is, unmodified, since we
   never edit vendored content) contains rules like `data/mimic/*` and
   `ANCE/` that were written for the *original* authors' local workflow.
   Left as default, these would have dropped the ANCE retriever source code
   (`train.py`, `data.py`, `multi_model.py`, `utils.py`,
   `gen_embeddings.py`, two `.sh` scripts) and the three schema-placeholder
   `data/mimic/*.json` files entirely. These were explicitly `git add -f`'d
   since they are real source code / harmless 1-record schema examples, not
   datasets. The genuinely empty checkpoint and embedding placeholder files
   (`*.pt`, `embedding/embedding`, `retrieval/retrieved_reports.json`) were
   left untracked, consistent with the project rule to never commit
   checkpoints — they carry zero information (0 bytes) either way.

## Licensing Note

The official repository is MIT-licensed, which permits adaptation. This
project still keeps the vendored copy under `reference/original_repository/`
untouched and writes all adapted/rewritten code under `src/baseline/`, per
project rules — this is a stricter separation than the license requires, not
one it requires.
