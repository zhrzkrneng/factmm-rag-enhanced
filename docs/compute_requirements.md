# Compute Requirements

## Evidence From the Official Repository

`OFFICIAL_REPOSITORY` (`src/retriever/DPR/train.sh` SLURM header):

```
--cpus-per-task=8
--gres=gpu:A6000:1     # single NVIDIA RTX A6000 (48GB)
--mem=60G
--time=1-00:00:00      # up to 24h for retriever (DPR-stage) training
```

The ANCE-stage `train.sh` has no SLURM header in the shipped copy but reuses
the same `train.py` entry point, so similar GPU memory needs likely apply
(`REASONABLE_INFERENCE`).

Factual-similarity scoring (`gen_similarity.sh`, `gen_topk_pos.sh`) is
CPU-only, chunked into 64 SLURM array tasks, each requesting 16G/12G RAM and
1 CPU — i.e., designed for a cluster with many small parallel jobs, not a
single machine, when run at full MIMIC-CXR scale (125,417² pairwise
comparisons before chunking).

## Evidence From the Paper (`PAPER_EXPLICIT`, Appendix A.1–A.2)

The paper states exact measured hardware/runtime, resolving what was
previously `UNKNOWN` from the code alone:

- **Retriever training**: 1x NVIDIA RTX A6000, **~10 hours** total
  (matches the SLURM header's 24h allowance above, with headroom).
- **RAG generator fine-tuning (LLaVA-1.5)**: **8x NVIDIA RTX A6000, ~4
  hours**, 1 epoch, learning rate 2e-5, global batch size 128, from a
  `vicuna-7b-v1.5` checkpoint. This is a materially larger requirement than
  the retriever stage and was not visible anywhere in the shipped code
  (no SLURM header exists in `train_llava.sh` in this clone) — treat the
  8-GPU figure as authoritative for planning purposes going forward.

## Implications for This Project

| Stage | Full-scale (paper-level) | Small-scale MVP (this project, Phase 3) |
|---|---|---|
| RadGraph/CheXbert annotation | GPU recommended for CheXbert; CPU-feasible but slow | CPU, tiny synthetic set, or mock annotator |
| Factual similarity scoring | O(n²) pairs, needs chunked/parallel execution at n≈125k | O(n²) with n≈10-50 synthetic records, trivial on CPU |
| Retriever training | 1x A6000 (48GB)-class GPU, ~10h measured (paper) | CPU or single small GPU, a handful of steps, correctness-only |
| Retriever ANCE stage | Similar to DPR stage plus hard-negative mining pass | Same reduced scale |
| Generator (LLaVA) fine-tune | **8x A6000, ~4h** (paper, confirmed) — well beyond a single-GPU Colab session | Mock generator only; real LLaVA fine-tune explicitly out of scope for interactive/Colab work unless the user has access to a comparable multi-GPU cluster and confirms it |
| Evaluation | GPU helpful for BERTScore/F1CheXbert/F1RadGraph at scale | CPU-feasible at small scale |

## Colab / Local Environment Notes

- A **free/standard Colab GPU runtime** (e.g., a single T4/A100 depending on
  tier) is plausible for: small-scale retriever fine-tuning steps, RadGraph/
  CheXbert annotation of small batches, and running the full evaluation
  suite on a handful of examples.
- A standard Colab runtime is **not** sufficient for full-corpus factual
  pair mining (125,417² comparisons) or the confirmed 8x-A6000 LLaVA
  fine-tuning stage — these need heavy sub-sampling, precomputed
  artifacts, or an external multi-GPU cluster. This is now a confirmed
  constraint (paper-stated hardware), not a guess.
- Two separate Python environments are required (see
  `docs/paper_analysis.md` §18): a retriever-stage environment
  (`torch==1.13.1`, `transformers==4.23.1`) and a generator-stage LLaVA
  environment (`transformers==4.36.2`, `peft==0.10.0`). Do not attempt to
  install both simultaneously in one environment.

## Hardware/Runtime Recording Requirement

Per project rules, every executed experiment must record: device type
(CPU/GPU model), wall-clock runtime, peak memory, and package versions,
in the experiment's result JSON (Milestone 2.6 onward). This is a
process requirement for *later* milestones, noted here for completeness —
no experiments have been run yet in Phase 0.
