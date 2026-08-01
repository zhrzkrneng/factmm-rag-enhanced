# Paper Analysis: FactMM-RAG

## Paper Access Status

**`paper/factmm_rag.pdf` is now available** (uploaded by the user directly
to the repository's `main` branch and merged into this branch). This
document has been rewritten against the actual paper text
(arXiv:2407.15268v2, "Fact-Aware Multimodal Retrieval Augmentation for
Accurate Medical Radiology Report Generation," Liwen Sun*, James Zhao*,
Megan Han, Chenyan Xiong — Carnegie Mellon University; *equal contribution).
Claims are now labeled `PAPER_EXPLICIT` where directly stated in the text,
`OFFICIAL_REPOSITORY` where sourced from the cloned code, and flagged
explicitly wherever the two sources **disagree** — this is the most
important output of this revision, since silently trusting either source
alone would hide real discrepancies.

## 1. Research Problem (`PAPER_EXPLICIT`)

Multimodal foundation models show promise for automating chest radiology
report generation, but suffer from hallucinated, factually inaccurate
output. Retrieval-Augmented Generation (RAG) can ground generation in
retrieved evidence, but building a medical multimodal retriever is hard
because it must bridge symptomatic image semantics and factually-equivalent
report text. Prior medical multimodal retrievers (GLoRIA, MedCLIP,
CXR-CLIP, BiomedCLIP, etc.) "neglect specific image information and do not
adequately emphasize factual accuracy, resulting in imprecision when
retrieving radiology reports."

## 2. Claimed Contributions (`PAPER_EXPLICIT`, verbatim from Introduction)

1. A fact-aware medical multimodal retriever to augment multimodal
   foundation models in generating accurate chest X-ray radiology reports.
2. A method for mining factually-informed radiology report pairs that
   trains multimodal encoders to retrieve high-quality reference reports.
3. A demonstration that, on two benchmark datasets, this retriever
   outperforms state-of-the-art medical multimodal retrievers on both
   language generation and clinically relevant metrics.

## 3. Complete Pipeline (`PAPER_EXPLICIT`, Section 3, cross-checked against code)

1. **Chest radiograph annotation**: RadGraph extracts structured entities
   (e.g., carina, lungs, abnormalities) and clinical relations (e.g.,
   modify, located at, suggestive of) from each report's free text, stored
   as `[(entity, label, relation), ...]` per report.
2. **Factual report pair mining**:
   - First, restrict candidates to reports sharing "the same symptom"
     (diagnostic label) as the query, to reduce false negatives.
   - Then compute a RadGraph-based factual similarity between query and
     candidate report (Eq. 1):
     `s(q,d) = 2·|q̂ ∩ d̂| / (len(q̂) + len(d̂))`, where `q̂`/`d̂` are the
     RadGraph-annotated (entity+relation) forms of the two reports — this
     is the standard F1-style RadGraph reward.
   - Keep candidates with `s(q,d) > δ` as positive pairs `N_q` (Eq. 2).
   - **This exactly matches the official code's two-part mask**
     (`chexbert_sim >= chex_thresh` AND `radgraph_sim >= radg_thresh`):
     the paper's "same symptom" pre-filter corresponds to the code's
     `chex_thresh=1.0` (exact 5-label match), and the paper's `δ`
     corresponds to the code's `radg_thresh` (`0.4` in shipped scripts).
     What looked like an unexplained combined-threshold design in the code
     audit is confirmed by the paper's narrative.
3. **Multimodal dense retrieval training**: a single universal encoder,
   **MARVEL** (Zhou et al. 2024), encodes the query image alone
   (`q = MARVEL(q_img)`, Eq. 3) and each candidate as an image+text pair
   (`d = MARVEL(d_txt, d_img)`, Eq. 4). Relevance is cosine similarity
   (Eq. 5). Trained with an InfoNCE-style contrastive loss (Eq. 6) using
   mined positives `d+` and in-batch negatives `d-` (Karpukhin et al. 2020,
   DPR-style), with temperature `τ`.
4. **Retrieval-augmented generation**: encode the query image, retrieve the
   **single highest-relevance** report from the training corpus, and pass
   image + retrieved report into a multimodal foundation model (LLaVA),
   fine-tuned with standard autoregressive language-modeling loss (Eq. 7)
   over the ground-truth report.

## 4. Datasets (`PAPER_EXPLICIT`)

- **MIMIC-CXR** (Johnson et al. 2019), processed per Delbrouck et al. 2023
  (the vilmedic ACL-2023 RadSum23 split): **125,417 training**, **991
  validation**, **1,624 test** image-report pairs, from Beth Israel
  Deaconess Medical Center. Used to train **both** the retriever and the
  foundation model.
- **CheXpert** (Irvin et al. 2019), Stanford Health Care: used **only** for
  **zero-shot evaluation** — 1,000 test pairs (the "hidden test set" from
  MIMIC-CXR-RRS, with CheXpert images downloaded separately from Stanford
  AIMI). This confirms what was previously `REASONABLE_INFERENCE` — the
  official repo ships only a CheXpert `test.json` placeholder because
  CheXpert is genuinely evaluation-only, never a training source.

## 5. Preprocessing (`PAPER_EXPLICIT`)

- Frontal view is explicitly selected ("Since each radiology study contains
  multiple image views for each patient, we select the frontal view") —
  confirms the `image[0]` convention seen in code is intentional, not
  incidental (previously flagged as an open question — now resolved).
- Finding and impression sections are concatenated to form the report text
  used for the retriever's candidate/text side.

## 6. RadGraph Annotation (`PAPER_EXPLICIT` + `OFFICIAL_REPOSITORY`)

RadGraph performs NER + relation extraction to build the structured
entity/relation representation described above. The paper does not name an
exact RadGraph package version; the official code pins `radgraph==0.0.9`
(`OFFICIAL_REPOSITORY`) — treat this pin as the operative version for
reproduction, per project rules against inventing missing detail.

## 7. Factual Pair Mining — Thresholds

- **Discrepancy found (flag for `docs/risk_register.md`)**: the paper's
  Implementation Details state "we rerank the retrieved reports by factual
  similarity and use the **top 2** factual report pairs for each query to
  train our multimodal retriever" (`PAPER_EXPLICIT`). The shipped
  `gen_topk_pos.sh` / `train.sh` scripts in the official repository use
  **`top_k=3`** (`OFFICIAL_REPOSITORY`). This is a real, unresolved
  mismatch between the paper text and the released code's default
  configuration — not something to silently pick one side of. This
  project's config will expose `top_k` as a parameter and run both values,
  documenting which one reproduces paper-reported numbers more closely.
- Threshold values `chexbert_threshold=1.0`, `radgraph_threshold=0.4` match
  between paper's Figure 2d (F1CheXbert threshold = 1, which the paper
  shows is the best-performing setting among {0, 0.4, 0.8, 1}) and the
  code's shipped defaults — `PAPER_EXPLICIT` + `OFFICIAL_REPOSITORY`,
  consistent.
- The paper explicitly studies threshold sensitivity as an ablation
  (Section 5.3, Figures 2–3): sweeping F1CheXbert ∈ {0, 0.4, 0.8, 1} and
  F1RadGraph ∈ {0.2, 0.3, 0.4, 0.5, 0.6}. Finding: stricter thresholds
  saturate/plateau and can exclude too many useful pairs; F1RadGraph
  threshold alone (without any CheXbert label filtering) can also mine
  effective pairs, showing the method doesn't strictly require diagnostic
  label supervision.

## 8. Retriever Architecture (`PAPER_EXPLICIT` + `OFFICIAL_REPOSITORY`)

- Backbone: **MARVEL** (Zhou et al. 2024), built on **T5-ANCE** (text) and
  a **Vision Transformer** (Dosovitskiy et al. 2021). The paper describes
  the vision encoder generically as "a vision transformer"; the official
  code's specific choice is `CLIPVisionModel`
  (`openai/clip-vit-base-patch32`) — a ViT-B/32 — which is consistent, just
  more specific than the paper's wording.
- Two MARVEL initialization checkpoints are compared: **WebQA** and
  **ClueWeb** (both from Zhou et al. 2024's MARVEL release); ClueWeb gives
  a marginal edge, attributed to larger pretraining scale
  (`PAPER_EXPLICIT`, Section 5.2). The headline "FactMM-RAG" numbers in
  Table 1 correspond to the **ClueWeb** checkpoint + LLaVA-1.5 (confirmed
  by matching values against the `ClueWeb-LLaVA1.5` row in Table 2).
- **Loss temperature discrepancy (flag)**: the paper's Appendix A.1 states
  a fixed temperature hyperparameter **τ = 0.01** for the contrastive loss
  (Eq. 6). The official code's `train.py` instead uses a **learned**
  `logit_scale` parameter initialized from a pretrained `CLIPModel`
  (typically initialized near `ln(1/0.07)`, i.e. an effective starting
  temperature around 0.07, not 0.01, and trainable rather than fixed). This
  is a genuine, currently unresolved discrepancy between the paper's stated
  hyperparameter and the released training script's actual mechanism —
  flagged in the risk register, not silently resolved in either direction.

## 9. Generator Architecture (`PAPER_EXPLICIT`)

- **LLaVA-1.5**, initialized from a **vicuna-7b-v1.5** checkpoint (Appendix
  A.2). LLaVA-1.6 is also tested as a backbone-variation ablation (Table 2,
  `ClueWeb-LLaVA1.6`), with similar performance to 1.5.
- Prompt templates (Figure 5, `PAPER_EXPLICIT`, matches `build_rag_dataset.py`):
  - Non-RAG (VQA): `Generate a radiology report from this image:\n<image>`
  - RAG: `Here is a report of a related patient:\n"<document>"\nGenerate a radiology report from this image:\n<image>`

## 10. Training Objectives & Hyperparameters (`PAPER_EXPLICIT`, Appendix A.1–A.2)

**Retriever training** (matches `train.py` defaults exactly, giving high
confidence both sources describe the same run):
AdamW optimizer, `epochs=15`, `early_stop=5`, `batch_size=32`,
`learning_rate=5e-6`, contrastive temperature `τ=0.01` (see discrepancy in
§8), modality-balanced hard negatives added after the in-batch-negative
stage (ANCE-style second stage), trained on **1x NVIDIA RTX A6000 for ~10
hours**.

**RAG fine-tuning (LLaVA)** — this fills a major gap the official code left
unspecified: `epochs=1`, `learning_rate=2e-5`, `global_batch_size=128`,
initialized from `vicuna-7b-v1.5`, trained on **8x NVIDIA RTX A6000 for ~4
hours**. Checkpoint saved after one full pass for final evaluation.

## 11. Negative Sampling (`PAPER_EXPLICIT` + `OFFICIAL_REPOSITORY`)

Two stages, confirming the earlier code-only reading: (1) in-batch
negatives during the initial dense-retrieval training stage; (2)
modality-balanced **hard negatives**, following Yu et al. 2023a / Zhou et
al. 2024 (MARVEL's own hard-negative recipe) — the paper does not restate
the code's specific hard-negative mining thresholds
(`chexbert_threshold=1`, `radgraph_threshold=0.4`, `topN=100`,
`num_top_neg=2`), so those numeric specifics remain sourced only from
`OFFICIAL_REPOSITORY`.

## 12. Retrieval Strategy at Inference (`PAPER_EXPLICIT`)

Exactly one report retrieved per query image (highest cosine-similarity
match), confirming the baseline never does multi-report retrieval — this
is squarely what Innovation A (adaptive multi-report retrieval) extends
beyond.

## 13. Generation Strategy (`PAPER_EXPLICIT`)

Single retrieved report is embedded into the fixed RAG prompt template
(§9) alongside the query image; LLaVA generates the report autoregressively.

## 14. Baselines (`PAPER_EXPLICIT`, Section 4 "Baselines")

- **Multimodal retrievers compared**: CLIP (Radford et al. 2021, general-
  domain), GLoRIA (Huang et al. 2021), MedCLIP (Wang et al. 2022), CXR-CLIP
  (You et al. 2023), BiomedCLIP (Zhang et al. 2024), Med-MARVEL (MARVEL
  contrastively trained on each patient's *own* image-report pair, without
  factual pair mining — the direct ablation target for isolating the
  contribution of factual pair mining).
- **Non-RAG comparisons**: "No Retriever" (direct LLaVA fine-tune, no
  retrieval); ORGan (Hou et al. 2023, observation-plan + tree reasoning).
- **Upper bound**: Oracle — retrieves the training-corpus report maximizing
  `F1RadGraph + F1CheXbert` instance-wise similarity to the query
  (excluding self for training queries; no such exclusion needed at test
  time since the corpus is the training set) — see Appendix A.3 for the
  exact `argmax` definition. This Oracle is **the paper's own upper-bound
  baseline**, not a project invention — our reproduction matrix and
  experiment design should reuse this exact definition rather than
  re-deriving an oracle procedure.

## 15. Evaluation Metrics (`PAPER_EXPLICIT`, Section 4 + Appendix A.3)

- **ROUGE-L** (longest common subsequence F-measure) and **BERTScore**
  (semantic similarity) for language fluency.
- **F1CheXbert**: micro-averaged F1 over 5 CheXbert-labeled observations
  (Cardiomegaly, Edema, Consolidation, Atelectasis, Pleural Effusion) —
  dataset-level only; for instance-level scores (used during pair mining),
  the paper uses the raw proportion of matching predicted classes
  (`np.sum(ref==hyp)/5`, values in `{0, 0.2, 0.4, 0.6, 0.8, 1.0}`), matching
  the code's `chexbert_similarity` function exactly.
- **F1RadGraph**: instance-level `RG_ER` reward
  (`reward_level="partial"` in the `radgraph` package), following the
  MIMIC-CXR-RRS convention — matches `src/evaluation.py`'s default exactly.
- **Statistical significance**: Table 1's caption states "FactMM-RAG
  outperforms the best baseline with p-value < 0.05" — so the paper
  **does** perform a significance test, contrary to what was assumed in
  the original code-only audit. The paper does **not** specify which test
  (paired t-test, bootstrap, permutation, etc.), so the exact procedure is
  still `UNKNOWN` and must be treated as our own methodological choice to
  document, not a reproduction of a specified test. Confidence intervals
  are not reported at all in the paper — our bootstrap-CI plan
  (Milestone 2.6) remains a genuine `PROPOSED_EXTENSION` beyond what the
  paper provides, but the significance-testing half of that milestone is
  now known to have *some* paper precedent, just an unspecified one.
- BLEU-4 is computed in the official `evaluation.py` script but is **not**
  reported anywhere in the paper's tables — treat BLEU-4 as an
  `OFFICIAL_REPOSITORY`-only extra metric, not a paper-reported one.

## 16. Ablation Studies (`PAPER_EXPLICIT`, Section 5.2–5.4)

1. **Multimodal-retrieval-only setting** (Table 2, top): evaluate the
   retriever alone (nearest report to test image vs. ground truth), without
   running it through the generator — isolates retrieval quality from
   generation quality.
2. **Backbone variation** (Table 2, bottom): MARVEL init from WebQA vs.
   ClueWeb checkpoints, and Med-MARVEL as an alternative retriever
   backbone; LLaVA-1.5 vs. LLaVA-1.6 as the generator.
3. **Fact-aware capability control** (Section 5.3, Figures 2–3): sweep
   F1CheXbert/F1RadGraph mining thresholds; show F1RadGraph alone
   (no CheXbert label filtering) still yields useful supervision.
4. **Fact-aware capability propagation** (Section 5.4, Figure 4): tracks
   MRR alongside F1CheXbert/F1RadGraph across training checkpoints,
   showing retrieval quality gains propagate into generation quality gains.
5. **Case studies** (Section 5.5, Tables 3–4): qualitative comparison of
   FactMM-RAG vs. Med-MARVEL generated/retrieved reports against ground
   truth, with per-example F1RadGraph and CheXbert-observation annotations.

## 17. Reported Results (`PAPER_EXPLICIT`, Table 1 — the reproduction target)

| Dataset | Model | F1CheXbert | F1RadGraph | ROUGE-L | BERTScore |
|---|---|---|---|---|---|
| MIMIC-CXR | No Retriever | 0.496 | 0.234 | 0.294 | 0.549 |
| MIMIC-CXR | ORGan | 0.541 | 0.240 | 0.308 | 0.552 |
| MIMIC-CXR | CLIP | 0.507 | 0.241 | 0.300 | 0.552 |
| MIMIC-CXR | GLoRIA | 0.476 | 0.232 | 0.294 | 0.543 |
| MIMIC-CXR | MedCLIP | 0.517 | 0.238 | 0.298 | 0.549 |
| MIMIC-CXR | CXR-CLIP | 0.501 | 0.243 | 0.302 | 0.553 |
| MIMIC-CXR | BiomedCLIP | 0.502 | 0.233 | 0.293 | 0.546 |
| MIMIC-CXR | Med-MARVEL | 0.537 | 0.237 | 0.306 | 0.549 |
| MIMIC-CXR | **FactMM-RAG** | **0.602** | **0.257** | 0.307 | **0.561** |
| MIMIC-CXR | Oracle | 0.972 | 0.523 | 0.495 | 0.677 |
| CheXpert (zero-shot) | No Retriever | 0.371 | 0.173 | 0.231 | 0.469 |
| CheXpert (zero-shot) | ORGan | 0.431 | 0.181 | 0.232 | 0.470 |
| CheXpert (zero-shot) | CLIP | 0.381 | 0.172 | 0.231 | 0.468 |
| CheXpert (zero-shot) | GLoRIA | 0.397 | 0.173 | 0.231 | 0.468 |
| CheXpert (zero-shot) | MedCLIP | 0.408 | 0.182 | 0.238 | 0.471 |
| CheXpert (zero-shot) | CXR-CLIP | 0.406 | 0.183 | 0.241 | 0.471 |
| CheXpert (zero-shot) | BiomedCLIP | 0.380 | 0.173 | 0.232 | 0.469 |
| CheXpert (zero-shot) | Med-MARVEL | 0.454 | 0.185 | 0.243 | 0.472 |
| CheXpert (zero-shot) | **FactMM-RAG** | **0.475** | 0.185 | 0.236 | **0.475** |
| CheXpert (zero-shot) | Oracle | 0.951 | 0.384 | 0.350 | 0.548 |

Retrieval-only setting (Table 2 top) and the 4 backbone-variation RAG
configurations (Table 2 bottom) contain further comparison numbers; see the
PDF directly for full precision when scoring our own reproduction against
it.

## 18. Implementation Details (`PAPER_EXPLICIT` + `OFFICIAL_REPOSITORY`)

- Retriever: 1x NVIDIA RTX A6000, ~10 hours (paper).
- RAG generator fine-tune: 8x NVIDIA RTX A6000, ~4 hours, 1 epoch (paper) —
  this is a materially larger compute requirement than anything visible in
  the code alone, and updates `docs/compute_requirements.md`.
- Environment pins (retriever-stage): `torch==1.13.1`,
  `transformers==4.23.1`, `radgraph==0.0.9`, `f1chexbert==0.0.2`
  (`OFFICIAL_REPOSITORY` — the paper itself gives no package versions).

## 19. Missing or Ambiguous Details (updated)

- **Resolved by the paper** (previously `UNKNOWN`, now `PAPER_EXPLICIT`):
  dataset split sizes, CheXpert's zero-shot-only role, frontal-view
  rationale, full baseline list, Oracle definition, generator hardware,
  headline result numbers, presence (but not exact method) of significance
  testing.
- **Newly surfaced discrepancies** (paper vs. shipped code, both read in
  full — see §7 and §8): `top_k` positives per query (paper: 2, code: 3);
  contrastive temperature (paper: fixed τ=0.01, code: learned
  `logit_scale`). These are not resolved by reading either source alone and
  are logged in `docs/risk_register.md`.
- **Still unknown**: exact RadGraph package/model version used by the
  paper's authors (only the code's `0.0.9` pin is known); exact
  significance-test procedure; per-checkpoint numeric values behind Figures
  2–4 (only described qualitatively/graphically in the PDF, not tabulated).

## 20. Limitations (`PAPER_EXPLICIT`, Section 7, near-verbatim)

- Scope is limited to chest radiology; not validated on other modalities
  (e.g., brain scans, histology).
- F1RadGraph/F1CheXbert don't capture report conciseness/clarity; no
  evaluation directly aligned with human judgment or domain-expert review
  of the pair-mining and evaluation procedure itself.
- No long-tail evaluation using more fine-grained ground-truth labels.

## 21. Ethics (`PAPER_EXPLICIT`, Section 8)

MIMIC-CXR is de-identified, credentialed data; the authors completed a
training course and signed a data use agreement; MIMIC-CXR's usage policy
prohibits sharing access with third parties. This directly reinforces
`docs/data_requirements.md`'s constraint that this project must never
attempt automated or shared access to MIMIC-CXR.

---

## Source Labels Used

- `PAPER_EXPLICIT` — directly stated in the paper text (now populated —
  paper is available)
- `OFFICIAL_REPOSITORY` — verified by reading the cloned official code/README
- `REASONABLE_INFERENCE` — inferred with explicit reasoning, not directly stated by either source
- `PROPOSED_EXTENSION` — not in paper or official code; our own addition
- `UNKNOWN` — genuinely unknown even with both sources now available
