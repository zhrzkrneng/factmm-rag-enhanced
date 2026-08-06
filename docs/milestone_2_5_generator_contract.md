# Milestone 2.5 — Retrieval-Augmented Generator: Final Implementation Contract

> **DESIGN CONTRACT — NOT EXECUTABLE CODE.** Every class, function, and
> config shown below is a signature-level specification for Cells 24–29,
> not implemented source. No file under `src/` is created or modified by
> this document. This is the single, consolidated, approved contract —
> it supersedes and merges the two chat-turn design documents that
> preceded it (the source audit + initial contract, and the four-item
> addendum); nothing here should be read alongside those as a separate
> or additional source of truth.

## Table of Contents

1. [Status & Scope](#1-status--scope)
2. [Preserved Source-Audit Findings](#2-preserved-source-audit-findings)
3. [Architecture Overview](#3-architecture-overview)
4. [Strict Patient Isolation Policy](#4-strict-patient-isolation-policy)
5. [PromptMode & PromptBuilder Contract](#5-promptmode--promptbuilder-contract)
   - 5.1 [`PromptBuilderConfig`](#51-promptbuilderconfig)
   - 5.2 [`PromptResult`](#52-promptresult)
   - 5.3 [Exact Prompt Strings](#53-exact-prompt-strings)
6. [`GeneratorConfig` & `GenerationConfig`](#6-generatorconfig--generationconfig)
7. [Module Layout & Responsibility Boundaries](#7-module-layout--responsibility-boundaries)
8. [Adapter Architecture](#8-adapter-architecture)
   - 8.1 [`GeneratorAdapter` (interface)](#81-generatoradapter-interface)
   - 8.2 [`PreparedGeneratorInputs`](#82-preparedgeneratorinputs)
   - 8.3 [`GenerationMetadata`](#83-generationmetadata)
   - 8.4 [`GeneratorResult`](#84-generatorresult)
   - 8.5 [`HFGeneratorAdapter`](#85-hfgeneratoradapter)
   - 8.6 [`MockGeneratorAdapter`](#86-mockgeneratoradapter)
9. [RAG Dataset Builder](#9-rag-dataset-builder)
   - 9.1 [`RAGDatasetBuilderConfig`](#91-ragdatasetbuilderconfig)
   - 9.2 [`RAGDatasetRow`](#92-ragdatasetrow)
   - 9.3 [Selection / Exclusion Behavior](#93-selection--exclusion-behavior)
10. [Output Schema (JSONL)](#10-output-schema-jsonl)
11. [Testing Strategy](#11-testing-strategy)
12. [Colab Execution Plan (Cells 24–29)](#12-colab-execution-plan-cells-2429)
13. [Known Risks & Paper/Code Discrepancies](#13-known-risks--papercode-discrepancies)
14. [Open Questions (`UNKNOWN`)](#14-open-questions-unknown)

---

## 1. Status & Scope

This is Milestone 2.5's approved design, produced through: (a) a source
audit of `paper/factmm_rag.pdf`, `docs/paper_analysis.md`,
`docs/reproduction_matrix.md`, `docs/risk_register.md`, and every
official generator-related file in
`reference/original_repository/FactMM-RAG/` (`install_llava.sh`,
`src/generator/*.py`, `src/generator/vqa/*.sh`); (b) an initial
implementation contract; (c) a four-item revision (strict patient
isolation, unified prompt API, split configs, separated modules, prompt
versioning, explicit seed, adapter abstraction, dedicated real-model
cell); (d) a four-item addendum (`PromptBuilderConfig`,
`GenerationMetadata`, `prepare_inputs`, `GeneratorResult`). All of that
is merged into one coherent contract below — nothing from the prior
two turns is dropped, and the reconciliations needed to merge them
without contradiction are called out explicitly in the response that
accompanies this file.

**Baseline only.** No Generator/Evaluation/Innovation code has been
written. No Colab cell has been created. No real LLaVA checkpoint has
been downloaded. Cell 24 begins only after this document is approved.

---

## 2. Preserved Source-Audit Findings

| Detail | Value | Label |
|---|---|---|
| Framework | LLaVA-1.5, pinned `haotian-liu/LLaVA` commit `c121f0432da27facab705978f83c4ada465e46fd` | `PAPER_EXPLICIT` + `OFFICIAL_REPOSITORY` |
| Base LM | `lmsys/vicuna-7b-v1.5` | `PAPER_EXPLICIT` + `OFFICIAL_REPOSITORY` |
| Vision encoder | `openai/clip-vit-large-patch14-336` — **a different CLIP instance from the Retriever's** `openai/clip-vit-base-patch32` (Milestone 2.4) | `OFFICIAL_REPOSITORY` |
| Vision layer | Second-to-last hidden layer (`--mm_vision_select_layer -2`) | `OFFICIAL_REPOSITORY` |
| Projector | `mlp2x_gelu`, warm-started from an unset `$PROJECTOR_PATH` shell variable (by convention the public `llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5` artifact) | `OFFICIAL_REPOSITORY` (mechanism); `REASONABLE_INFERENCE` (exact artifact identity) |
| Image token scheme | Single `<image>` placeholder (`mm_use_im_start_end=False`, `mm_use_im_patch_token=False`) — **architecturally distinct from the Retriever's per-patch `<im_start>/<im_patch>*N/<im_end>` splicing**; no shared special-token machinery between the two milestones | `OFFICIAL_REPOSITORY` |
| Trainable vs. frozen | Full fine-tune of LM + projector — no `--lora_enable`/`--lora_r` flags in either shipped train script (`train_llava.sh`, `vqa/train_llava_vqa.sh`); vision tower conventionally frozen | `OFFICIAL_REPOSITORY` (no-LoRA-flags: certain); `REASONABLE_INFERENCE` (frozen vision tower — LLaVA's own `train.py` is not vendored in this repo, not independently re-read) |
| `peft==0.10.0` | Installed by `install_llava.sh`, never invoked by either shipped train script | `OFFICIAL_REPOSITORY` (installed+unused); `UNKNOWN` (why) |
| Conversation template | `vicuna_v1` by name (`--version v1` train-side, `--conv-mode vicuna_v1` inference-side) | `OFFICIAL_REPOSITORY` (name only) |
| Conversation template exact string | System prompt / role separators — `llava/conversation.py` is not vendored in this repo (cloned externally at install time) and was not independently re-read in this sandbox (no network access) | `UNKNOWN` — general public knowledge of LLaVA-1.5's `vicuna_v1` template exists but is `REASONABLE_INFERENCE` at best, not verified against the exact pinned commit |
| Decoding (official) | `temperature=0` (greedy), no beam flag present (`num_beams=1` by omission) | `OFFICIAL_REPOSITORY` |
| Prompt strings: figure vs. code | `paper/factmm_rag.pdf` Figure 5's visual line breaks (confirmed via direct `pdftotext -layout` extraction) do **not** match `build_rag_dataset.py`/`build_nonrag_dataset.py`'s literal f-strings byte-for-byte (different whitespace/newline placement) | `PAPER_EXPLICIT` (figure, visual layout only) vs. `OFFICIAL_REPOSITORY` (literal string, the actual training-time ground truth) — genuine gap, not previously flagged in `docs/paper_analysis.md` |
| RAG-dataset short-text filter | Paper (Appendix A.2): "we filter out... malformed reports... specified by being less than 5 **characters**", worded as unconditional. Code: `--test_short` is an **opt-in** flag (`action="store_true"`, default `False`), and the actual check is `len(text.split()) < 5` — **words**, not characters | `PAPER_EXPLICIT` vs. `OFFICIAL_REPOSITORY` — contradicts `docs/reproduction_matrix.md`'s current "paper and code agree exactly" claim for the RAG-dataset-construction row; that row needs correction once this milestone is implemented (not done yet) |
| Official fallback-on-exhaustion | If no KNN candidate passes the active filters (self-study / self-patient / optional short-text), official code falls back to the **rank-0 candidate regardless of leakage**, logged as an "anomaly," never dropped from the dataset | `OFFICIAL_REPOSITORY` — this is exactly what §4 (Strict Patient Isolation) below replaces with a safe default, kept available opt-in for controlled reproduction |
| Retrieval strategy | Exactly one retrieved report per query, top-1 only, never multi-report | `PAPER_EXPLICIT` §12 |
| Split usage | Retrieval corpus is always train-only, even for valid/test queries (paper's Oracle description: "no such exclusion needed at test time since the corpus is the training set") | `PAPER_EXPLICIT` |
| Ground-truth field | Generator target is `finding` alone (`output_data_mode="finding"` default), **not** `combined_text` (the Retriever's own candidate representation, `finding + " " + impression`) | `OFFICIAL_REPOSITORY` (code default) |
| Full fine-tune hyperparameters | `epochs=1`, `lr=2e-5`, `global_batch_size=128`, `bf16=True`, `tf32=True`, cosine LR schedule, `warmup_ratio=0.03`, `gradient_checkpointing=True`, DeepSpeed ZeRO-3, 8× NVIDIA RTX A6000, ~4 hours | `PAPER_EXPLICIT` + `OFFICIAL_REPOSITORY` (both agree) |
| Full fine-tune memory | ~14GB weights (bf16) + ~84GB naive (non-sharded) AdamW optimizer state (7B params × ~12 bytes/param) + gradients ⇒ **~100GB+** total, explaining the ZeRO-3/8-GPU requirement | Estimate, not measured in this sandbox (no GPU available here) |
| LoRA MVP memory | ~14GB frozen base (bf16) + <1GB LoRA optimizer state + activation overhead ⇒ **~18–25GB** estimated | Estimate, not measured — general LoRA/QLoRA literature, unverified against an actual run |

---

## 3. Architecture Overview

```
                 ┌────────────────────┐
                 │  RAGDatasetBuilder │  (dataset_builder.py)
                 │  strict isolation  │
                 └─────────┬──────────┘
                            │ RAGDatasetRow
                            ▼
                 ┌────────────────────┐
                 │   PromptBuilder     │  (prompt_builder.py)
                 │   .build(mode, ...) │
                 └─────────┬──────────┘
                            │ PromptResult
                            ▼
                 ┌────────────────────┐
                 │   LLaVAGenerator    │  (generator.py) — thin orchestrator
                 │ (GenerationConfig)  │    only; no prompt literals, no
                 └─────────┬──────────┘    dataset-filter logic, no direct
                            │ depends only  transformers/torch imports
                            │ on interface
                            ▼
                 ┌────────────────────┐
                 │  GeneratorAdapter   │  (adapter.py — abstract interface)
                 │  .prepare_inputs()  │
                 │  .generate()        │
                 │  .generate_batch()  │
                 └──────┬──────┬──────┘
                         │      │
             ┌───────────┘      └───────────┐
             ▼                               ▼
   ┌───────────────────┐          ┌───────────────────┐
   │ HFGeneratorAdapter │          │ MockGeneratorAdapter│
   │ (adapter.py, real) │          │ (mock_adapter.py)   │
   │ constructed from   │          │ deterministic, no   │
   │ GeneratorConfig     │          │ ML dependency        │
   └───────────────────┘          └───────────────────┘
```

`LLaVAGenerator` never constructs an adapter itself — it is handed an
already-built `GeneratorAdapter` instance. `GeneratorConfig` is the
**adapter's own construction config** (consumed by
`HFGeneratorAdapter.__init__`/`MockGeneratorAdapter.__init__`), not a
parameter of `LLaVAGenerator`. `LLaVAGenerator` only needs the injected
adapter, a `PromptBuilder`, and a `GenerationConfig` (decoding
parameters). This is the one architectural clarification needed to
merge the two prior design turns without contradiction — see the
accompanying chat response for the full list of reconciliations.

---

## 4. Strict Patient Isolation Policy

Two explicit, mutually-exclusive flags on `RAGDatasetBuilderConfig`
(§9.1) replace the earlier, single `reproduce_official_fallback` idea:

```
strict_patient_isolation: bool = True   # project default
reproduce_official_bug: bool = False    # never the default
```

- **`strict_patient_isolation=True`** (default): self-study and
  self-patient candidates are rejected unconditionally, for every
  query, every split. If the KNN ranking is exhausted with no
  candidate passing every active filter, the query is **excluded**
  from the dataset — logged (`excluded_queries` count + per-query
  reason in the sidecar metadata), never silently included with a
  leaking candidate. One excluded query never aborts the whole run.
- **`reproduce_official_bug=True`**: reproduces the official code's
  exhaustion fallback (rank-0 candidate regardless of leakage)
  exactly, for controlled A/B reproduction experiments only.
- Setting both `True` simultaneously — or `reproduce_official_bug=True`
  while `strict_patient_isolation` is left at its default `True` — is
  a `ValueError` at config construction. The two are mutually
  exclusive, never silently reconciled or stacked; enabling the
  official-bug path requires the caller to explicitly also pass
  `strict_patient_isolation=False`.
- Both flags, and which one was actually active, are recorded per-row
  in `RAGDatasetRow` (§9.2) and in the builder's run-level sidecar
  metadata (mirroring `PairMiningConfig.as_actual_used_dict()`) — an
  experiment's provenance must be provable, not inferred.
- **Explicitly and permanently documented as an intentional deviation
  from `OFFICIAL_REPOSITORY` behavior.** The official leakage-permitting
  fallback is preserved in code (behind the opt-in flag), not deleted
  from the record — this project reproduces the discrepancy rather than
  erasing it, consistent with every other paper-vs-code gap in this
  contract.

---

## 5. PromptMode & PromptBuilder Contract

```python
class PromptMode(Enum):
    RAG_TRAIN = "rag_train"
    RAG_INFERENCE = "rag_inference"
    VQA_TRAIN = "vqa_train"
    VQA_INFERENCE = "vqa_inference"

class PromptBuilder:
    def __init__(self, config: PromptBuilderConfig) -> None: ...

    def build(
        self,
        mode: PromptMode,
        *,
        retrieved_report: Optional[str] = None,   # required for RAG_*, forbidden for VQA_*
        target_report: Optional[str] = None,      # required for *_TRAIN, forbidden for *_INFERENCE
    ) -> PromptResult:
        ...
```

**Pure and deterministic**: no I/O, no randomness, no hidden state
beyond the immutable `PromptBuilderConfig` — identical inputs always
produce a byte-identical `PromptResult`, independently unit-testable.
`build()` raises `ValueError` (never a silent default) when a
mode-required argument is missing or a mode-forbidden one is supplied.

### 5.1 `PromptBuilderConfig`

```python
@dataclass(frozen=True)
class PromptBuilderConfig:
    config_version: str = "1.0"
    prompt_version: str = "official_v1"        # stamped into every PromptResult; identifies
                                                 # which literal template set is active
    image_token: str = "<image>"                # the placeholder LLaVA's forward() splices
                                                 # projected patches into
    retrieved_report_quote_char: str = '"'       # wraps the retrieved report in RAG_* modes
    reject_empty_retrieved_report: bool = True   # RAG_*: raise ValueError on empty/whitespace-only
                                                  # retrieved_report rather than emit a malformed prompt
    max_retrieved_report_length: Optional[int] = None  # None = no limit (matches official code's
                                                  # own lack of truncation); when set, build() raises
                                                  # ValueError on overflow -- never silently truncates
```

`__post_init__` validates: non-empty `config_version`/`prompt_version`/
`image_token`; `retrieved_report_quote_char` exactly one character;
`max_retrieved_report_length` a positive int or `None`.

The four literal template shapes (§5.3) are **not** config fields —
they are internal, version-pinned string constants inside
`prompt_builder.py`, test-verified byte-for-byte against the official
code. `PromptBuilderConfig` governs versioning/metadata and a small set
of legitimately-variable, safety-relevant knobs; it never exposes the
core wording as mutable text, which would undermine exact reproduction.

### 5.2 `PromptResult`

```python
@dataclass(frozen=True)
class PromptResult:
    text: str                          # the inference-ready / instruction text
    conversations: Optional[list]      # the {"from": "human"/"gpt", "value": ...} turns,
                                        # populated only for *_TRAIN modes; None for *_INFERENCE
    prompt_version: str                # copied from PromptBuilderConfig.prompt_version
```

### 5.3 Exact Prompt Strings

Ground truth is the **official code's literal f-strings**, not Figure
5's visual line layout (see §2). Reproduced exactly, per `PromptMode`:

| Mode | Exact string |
|---|---|
| `RAG_TRAIN` (conversational) | `Here is a report of a related patient: "{retrieved_report}"\nGenerate a radiology report from this image:<image>` — plus a `{"from": "gpt", "value": target_report}` turn |
| `RAG_INFERENCE` | `Here is a report of a related patient: "{retrieved_report}"\nGenerate a radiology report from this image:` — no `<image>` literal; the adapter inserts the image from `image_path`, not from this text |
| `VQA_TRAIN` (conversational) | `Generate a radiology report from this image:<image>` — plus a `{"from": "gpt", "value": target_report}` turn |
| `VQA_INFERENCE` | `\nGenerate a radiology report from this image:` — note the **leading** `\n`, absent in `RAG_INFERENCE`; this small asymmetry exists in the official code itself, not introduced here |

No space or newline precedes `<image>` in either `*_TRAIN` string;
exactly one space (not newline) follows "patient:"; no normalization,
truncation, or casing change is ever applied to `retrieved_report`.

---

## 6. `GeneratorConfig` & `GenerationConfig`

Two separate frozen, validated dataclasses — deliberately mirroring
`transformers.GenerationConfig`'s own field philosophy where practical
(decoding parameters are orthogonal to model identity).
`GeneratorConfig` is the adapter's own construction config (see §3);
`GenerationConfig` is passed at call time to `generate()`/
`generate_batch()`.

```python
@dataclass(frozen=True)
class GeneratorConfig:
    """Adapter construction config -- what to load, never how to decode.
    Consumed by HFGeneratorAdapter.__init__ / MockGeneratorAdapter.__init__,
    not by LLaVAGenerator directly."""
    config_version: str = "1.0"
    llava_checkpoint: str = ...                     # base LLaVA checkpoint path/repo id
    base_lm_name: str = "lmsys/vicuna-7b-v1.5"
    vision_tower_name: str = "openai/clip-vit-large-patch14-336"
    adapter_kind: Literal["hf", "mock"] = "mock"    # descriptive/provenance only -- feeds
                                                      # GenerationMetadata.adapter_kind; actual
                                                      # dispatch is by which class is instantiated,
                                                      # not by reading this field
    lora_enabled: bool = False
    lora_checkpoint: Optional[str] = None
    conv_mode: str = "vicuna_v1"
    tokenizer_name: Optional[str] = None             # defaults to base_lm_name if None

@dataclass(frozen=True)
class GenerationConfig:
    """Decoding-time configuration -- how to decode, never what to load."""
    config_version: str = "1.0"
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_new_tokens: int = 256
    num_beams: int = 1
    seed: int = 42                # ALWAYS present, even at temperature=0 -- kept so that
                                    # enabling sampling later needs no schema migration, and
                                    # every run's provenance is fully reproducible regardless
                                    # of which decoding mode was actually used
    deterministic: bool = True    # explicit, never silently derived from temperature==0
```

`GeneratorConfig.__post_init__`: non-empty `config_version`/checkpoint
names; `adapter_kind` in `{"hf","mock"}`; `lora_checkpoint` required
when `lora_enabled=True` and forbidden otherwise (`ValueError`).

`GenerationConfig.__post_init__`: `temperature >= 0`; `top_p` in
`(0, 1]` or `None`; `top_k` positive or `None`; `max_new_tokens`/
`num_beams` positive ints; `seed` an int; `deterministic=True` combined
with `temperature > 0` is a `ValueError` — the two must agree.

---

## 7. Module Layout & Responsibility Boundaries

```
src/generation/
    __init__.py          # unchanged shared package docstring
    base.py               # Generator ABC (shared interface, mirrors src/retrieval/base.py)

src/baseline/generation/
    __init__.py
    prompt_builder.py     # PromptMode, PromptBuilder, PromptBuilderConfig, PromptResult
    dataset_builder.py    # RAGDatasetRow, RAGDatasetBuilder, RAGDatasetBuilderConfig
    adapter.py             # GeneratorAdapter (interface), PreparedGeneratorInputs,
                            #   GenerationMetadata, GeneratorResult, HFGeneratorAdapter (real)
    mock_adapter.py        # MockGeneratorAdapter (deterministic fake, no ML dependency)
    generator.py           # LLaVAGenerator (orchestrator), GeneratorConfig, GenerationConfig,
                            #   _generator_result_to_jsonl_row (pure JSONL mapping function)
```

This is a deliberate, disclosed reorganization of the Phase-1 scaffold:
the shared abstract `Generator` interface stays in `src/generation/
base.py` (matching the established `src/retrieval/base.py` vs.
`src/baseline/retrieval/*` split used project-wide), while every
concrete module — including `PromptBuilder` and the RAG dataset
builder, which the original stub docstrings had placed under the
shared `src/generation/` package — moves into `src/baseline/
generation/`.

**Responsibility boundaries** (strict): `generator.py` contains no
prompt-string literals and no dataset-filtering logic — those live
exclusively in `prompt_builder.py` and `dataset_builder.py`.
`generator.py` never imports `transformers`/`torch` at module scope for
model construction (only `adapter.py`'s `HFGeneratorAdapter` does,
lazily, inside private factory functions — the exact pattern already
established by `MultiModalRetriever`'s `_default_*_factory` functions).
`PreparedGeneratorInputs`/`GenerationMetadata`/`GeneratorResult` live in
`adapter.py` alongside the interface they belong to, not in
`generator.py`.

---

## 8. Adapter Architecture

Mirrors `MultiModalRetriever`'s dependency-injection discipline, but as
a fully-constructed object handed to `LLaVAGenerator`, not raw
factories consumed internally.

### 8.1 `GeneratorAdapter` (interface)

```python
class GeneratorAdapter(abc.ABC):
    @abc.abstractmethod
    def prepare_inputs(self, image_path: str, prompt_text: str) -> PreparedGeneratorInputs: ...

    @abc.abstractmethod
    def generate(self, image_path: str, prompt: str, generation_config: GenerationConfig) -> GeneratorResult: ...

    @abc.abstractmethod
    def generate_batch(
        self, requests: Sequence[Tuple[str, str]], generation_config: GenerationConfig
    ) -> List[GeneratorResult]: ...
```

`generate()`/`generate_batch()` return `GeneratorResult`/
`List[GeneratorResult]` directly (not a bare string) — this is the one
signature refinement needed to reconcile the addendum's `GeneratorResult`
introduction with the earlier draft, which had `generate()` return
`str`. A bare string cannot carry `GenerationMetadata` (checkpoint
revision, hardware, timestamp), which only the adapter has direct
knowledge of at generation time — so the adapter, not `generator.py`,
constructs the `GeneratorResult`.

**`prepare_inputs` responsibility**: the sole seam that turns a raw
`(image_path, prompt_text)` pair into fully model-ready tensors — real
image preprocessing (`HFGeneratorAdapter`: pad-to-square + the real
CLIP-ViT-L/14-336 processor) and real tokenization/conversation-template
application (`vicuna_v1` turn formatting around `prompt_text`).
`generate()`/`generate_batch()` call `prepare_inputs()` internally as
their first step, but it is exposed as its own **separately-testable
abstract method** — not a private helper — specifically so a test can
assert `MockGeneratorAdapter.prepare_inputs()` produces correct
shapes/placeholder alignment without invoking any generation logic, and
so `generator.py`'s orchestration tests can spy on
`adapter.prepare_inputs` being called with the exact prompt
`PromptBuilder.build()` produced. This mirrors the Retriever's own
decomposition exactly: `RetrievalCollator` (input preparation) is
separately testable from `MultiModalRetriever.encode_*` (the forward
pass) — `prepare_inputs` is that same seam, formalized on the adapter
interface.

### 8.2 `PreparedGeneratorInputs`

```python
@dataclass(frozen=True)
class PreparedGeneratorInputs:
    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
```

### 8.3 `GenerationMetadata`

Scoped strictly to **model/runtime provenance** — never duplicates
`GenerationConfig`'s decoding parameters, which are referenced
alongside it in the output row (§10), not copied into it.

```python
@dataclass(frozen=True)
class GenerationMetadata:
    adapter_kind: Literal["hf", "mock"]
    model_checkpoint: str                 # identifier/path/repo id actually loaded
    resolved_revision: Optional[str]      # real HF revision hash when resolved; None for mock
    base_lm_name: str
    vision_tower_name: str
    lora_checkpoint: Optional[str]
    conv_mode: str
    git_commit: Optional[str]             # reuses index.py's _resolve_git_commit() pattern
    torch_version: Optional[str]
    transformers_version: Optional[str]
    hardware: str                         # e.g. "cpu" / "cuda:0 (NVIDIA A100)", matching the
                                            # runtime-detection style from Milestone 2.4G
    generation_timestamp_utc: str         # reuses the project's existing _utc_now_iso() convention
```

Populated by the adapter at generation time: real values from
`HFGeneratorAdapter`; deterministic placeholder values (e.g.
`adapter_kind="mock"`, `resolved_revision=None`, `hardware="mock"`)
from `MockGeneratorAdapter` — identical shape across both adapters, no
adapter-dependent field is ever silently missing.

### 8.4 `GeneratorResult`

```python
@dataclass(frozen=True)
class GeneratorResult:
    query_key: QueryKey
    generated_report_text: Optional[str]   # None when error is set
    generation_metadata: GenerationMetadata
    error: Optional[str] = None
```

This **is** the generation outcome — nothing else in the pipeline
constructs one ad hoc. The output JSONL row (§10) is produced by one
explicit, pure, separately-unit-testable mapping function in
`generator.py`:

```python
def _generator_result_to_jsonl_row(
    result: GeneratorResult,
    *,
    retrieved_evidence_key: Optional[QueryKey],
    prompt_version: str,
    paper_version: str,
    implementation_version: str,
    generation_config: GenerationConfig,
) -> dict:
    ...
```

A JSONL row = `GeneratorResult` (the generation outcome) **plus** the
surrounding per-sample dataset/versioning context that is not part of
the outcome itself — composed by this one function, never duplicated
or diverged across call sites.

### 8.5 `HFGeneratorAdapter`

```python
class HFGeneratorAdapter(GeneratorAdapter):
    """Real adapter: constructs the real vision tower/tokenizer/base LM
    per GeneratorConfig, via the same all-factories-or-none injection
    pattern as MultiModalRetriever -- real transformers/torch calls are
    only ever attempted when factories are omitted."""
    def __init__(
        self,
        config: GeneratorConfig,
        *,
        model_factory: Optional[Callable[[], object]] = None,
        tokenizer_factory: Optional[Callable[[], object]] = None,
        image_processor_factory: Optional[Callable[[], object]] = None,
    ) -> None: ...
```

### 8.6 `MockGeneratorAdapter`

```python
class MockGeneratorAdapter(GeneratorAdapter):
    """Deterministic, template-filling fake -- no ML dependency at all.
    Used by every unit test and every synthetic smoke test (Cells 24-28)."""
```

**No unit test may construct an `HFGeneratorAdapter` with its real
factories.** Every `generator.py`/`dataset_builder.py`/
`prompt_builder.py` test uses `MockGeneratorAdapter` exclusively;
`HFGeneratorAdapter`'s real path is exercised only in Cell 29 (§12).

---

## 9. RAG Dataset Builder

### 9.1 `RAGDatasetBuilderConfig`

```python
@dataclass(frozen=True)
class RAGDatasetBuilderConfig:
    config_version: str = "1.0"
    rag_data_mode: str = "finding"        # field inserted into the prompt as retrieved evidence
    output_data_mode: str = "finding"     # field used as the generation target
    strict_patient_isolation: bool = True # see §4
    reproduce_official_bug: bool = False  # see §4
    test_short: bool = False              # kept OFF by default, matching official code's own
                                            # opt-in default -- never silently turned on
    min_short_words: int = 5              # matches official code's word-count threshold exactly
    corpus_scope: str = "train_only"      # purely descriptive, mirrors PairMiner's own field
```

`__post_init__` validates: non-empty `config_version`; `rag_data_mode`/
`output_data_mode` non-empty; `min_short_words` a positive int;
`strict_patient_isolation=True` combined with `reproduce_official_bug=
True` raises `ValueError` (see §4).

### 9.2 `RAGDatasetRow`

```python
@dataclass(frozen=True)
class RAGDatasetRow:
    query_key: QueryKey
    retrieved_key: Optional[QueryKey]        # None when excluded under strict isolation
    image_path: str
    retrieved_report_text: Optional[str]
    target_report_text: str
    rank_selected: Optional[int]             # which KNN rank was chosen; None when excluded
    num_candidates_rejected_self_study: int
    num_candidates_rejected_self_patient: int
    num_candidates_rejected_short_text: int
    fallback_used: bool                      # only ever True under reproduce_official_bug=True
    excluded: bool
    strict_patient_isolation: bool           # which mode was actually active for this row
    reproduce_official_bug: bool
    config_version: str
```

### 9.3 Selection / Exclusion Behavior

Iterates a query's KNN ranking (reusing `FaissFlatIPIndex.search()`'s
output shape from Milestone 2.4 — `List[List[QueryKey]]` + scores, not
a hand-rolled pickle format) looking for the first candidate that is
not the same study, not the same patient, and (only if `test_short=
True`) not under `min_short_words` words. Deterministic — the KNN
ranking itself is already deterministic (Milestone 2.4), and this
filter-and-pick-first-passing logic adds no randomness. On exhaustion:
`strict_patient_isolation=True` excludes the row (§4);
`reproduce_official_bug=True` falls back to the rank-0 candidate
exactly as official code does, setting `fallback_used=True`. Resume/
caching follows the exact established pattern from `PairMiner`/
`RadGraphAnnotator` (`_load_completed_*_keys`, sidecar `.meta.json`,
atomic writes, fail-fast vs. `continue_on_error`) — the fourth
implementation of this same pattern in this project, not a new one.
Corpus is always the train split (§2); a pre-flight assertion reusing
Milestone 2.1's `PatientSplitValidator` confirms the query split and
corpus split share no patients before any retrieval happens, failing
loudly if that invariant doesn't hold.

---

## 10. Output Schema (JSONL)

One row per generated sample, produced exclusively by
`_generator_result_to_jsonl_row` (§8.4):

```json
{
  "query_key": ["mimic-cxr", "p10000032", "s50414267"],
  "image_path": "...",
  "retrieved_evidence_key": ["mimic-cxr", "p10000200", "s50123456"],
  "prompt_version": "official_v1",
  "paper_version": "factmm_rag_naacl2025",
  "implementation_version": "1.0",
  "generated_report_text": "...",
  "generation_config": {
    "temperature": 0.0, "top_p": null, "top_k": null,
    "max_new_tokens": 256, "num_beams": 1, "seed": 42, "deterministic": true
  },
  "generation_metadata": {
    "adapter_kind": "hf", "model_checkpoint": "...", "resolved_revision": "...",
    "base_lm_name": "lmsys/vicuna-7b-v1.5",
    "vision_tower_name": "openai/clip-vit-large-patch14-336",
    "lora_checkpoint": null, "conv_mode": "vicuna_v1", "git_commit": "...",
    "torch_version": "...", "transformers_version": "...",
    "hardware": "cpu", "generation_timestamp_utc": "2026-08-02T00:00:00Z"
  },
  "error": null
}
```

Resume/atomic-write/`continue_on_error` behavior identical to every
prior JSONL-producing stage in this project (RadGraph annotation, pair
mining, retrieval dataset loaders).

---

## 11. Testing Strategy

- `PromptBuilder.build()` purity/determinism across all 4 `PromptMode`
  values, byte-compared against §5.3's exact strings.
- `build()`'s `ValueError`s for missing/forbidden arguments per mode.
- `PromptBuilderConfig` validation (empty strings, quote-char length,
  `max_retrieved_report_length` violations, empty-retrieved-report
  rejection).
- `GeneratorConfig`/`GenerationConfig` validated independently;
  `deterministic=True` + `temperature>0` rejected.
- `RAGDatasetBuilderConfig`'s `strict_patient_isolation`/
  `reproduce_official_bug` mutual-exclusivity `ValueError`.
- Same-study and same-patient candidates rejected even at KNN rank 0.
- Strict-mode exclusion vs. official-bug-reproduction fallback, both
  tested explicitly, never conflated.
- `GeneratorAdapter` interface conformance for `MockGeneratorAdapter`
  (full) and `HFGeneratorAdapter` (construction-only, injected fakes).
- `prepare_inputs()` independently testable: correct tensor shapes/
  placeholder alignment from `MockGeneratorAdapter`, without invoking
  generation.
- `generator.py` never imports `transformers`/`torch` at module scope
  for model construction (static/import-time check, mirroring
  `model.py`'s lazy default-factory pattern).
- `seed` present and round-trips into the output row even at
  `temperature=0`.
- `_generator_result_to_jsonl_row` unit-tested as a pure function:
  given a fixed `GeneratorResult` + context, produces the exact
  expected dict, including the error-path shape
  (`generated_report_text=None`, `error` populated).
- Output schema: required-field validation, duplicate-key rejection
  (same established style as every other loader in this project).
- Resume behavior: re-running skips completed keys, never duplicates
  output lines.
- One real-model forward/generation smoke test — reserved for Cell 29
  only (§12), never run as part of the Cell 24–28 unit test suite.

---

## 12. Colab Execution Plan (Cells 24–29)

- **Cell 24** — `prompt_builder.py` (`PromptMode`, `PromptBuilder`,
  `PromptBuilderConfig`, `PromptResult`) + unit tests.
- **Cell 25** — `dataset_builder.py` (`RAGDatasetRow`,
  `RAGDatasetBuilder`, `RAGDatasetBuilderConfig`, strict-isolation and
  official-bug-reproduction modes) + unit tests.
- **Cell 26** — `adapter.py`/`mock_adapter.py` (`GeneratorAdapter`,
  `PreparedGeneratorInputs`, `GenerationMetadata`, `GeneratorResult`,
  `HFGeneratorAdapter` skeleton with injected fakes only,
  `MockGeneratorAdapter`) + unit tests.
- **Cell 27** — `generator.py` (`LLaVAGenerator` orchestrator,
  `GeneratorConfig`, `GenerationConfig`,
  `_generator_result_to_jsonl_row`) + unit tests, `MockGeneratorAdapter`
  exclusively.
- **Cell 28** — synthetic end-to-end smoke test tying
  `RAGDatasetBuilder` → `PromptBuilder` → `MockGeneratorAdapter` →
  `LLaVAGenerator` → output-schema validation → resume, mirroring Cell
  23's full-lifecycle pattern.
- **Cell 29 — Real Generator Dry Run** (dedicated, mirrors Milestone
  2.4G exactly): real `HFGeneratorAdapter` construction (real
  CLIP-ViT-L/14-336, real `vicuna-7b-v1.5` tokenizer/LM, real
  `vicuna_v1` conversation template resolution), one synthetic image +
  one synthetic retrieved report through a real
  `PromptBuilder.build(PromptMode.RAG_INFERENCE, ...)` → real
  `adapter.generate(...)` call, classified error handling
  (`authentication_required`/`repository_unavailable`/`network_error`/
  `version_incompatibility`/`architecture_mismatch`, reusing the exact
  classifier from Milestone 2.4G, not reinvented), no training, no
  fine-tuning, no dataset download beyond the single synthetic sample,
  optional LoRA-adapter-load smoke test gated behind an explicit
  off-by-default flag (mirroring `ATTEMPT_FULL_CHECKPOINT_DOWNLOAD`
  from Cell 22/Milestone 2.4G). Ends in `CELL 29: PASS`/`FAIL`.

---

## 13. Known Risks & Paper/Code Discrepancies

- Exact LLaVA checkpoint/version: pinned commit
  `c121f0432da27facab705978f83c4ada465e46fd`, not vendored, not yet
  cloned/verified in this session.
- Base model license/access: `vicuna-7b-v1.5` license/gating status not
  independently re-verified from this sandbox (network-blocked);
  general knowledge only, `REASONABLE_INFERENCE`.
- GPU memory: full fine-tune ~100GB+ (needs ZeRO-3/8-GPU); LoRA MVP
  ~18–25GB estimated, unmeasured — no Colab-feasibility claim without
  an actual dry run.
- Prompt differences, paper figure vs. code (§2, §5.3) — not previously
  flagged in `docs/paper_analysis.md`.
- Image preprocessing version drift: Generator's CLIP
  (`clip-vit-large-patch14-336`) differs from the Retriever's CLIP
  (`clip-vit-base-patch32`) — different checkpoints, different
  preprocessing, never to be assumed shared.
- Retrieved-report leakage: official code's exhaustion fallback can
  leak a same-patient/same-study candidate — addressed by §4's strict
  default, with the official behavior preserved opt-in, not deleted.
- Full fine-tuning infeasibility on Colab/single-GPU hardware —
  confirmed, unchanged.
- `docs/reproduction_matrix.md`'s RAG-dataset-construction row
  currently claims "paper and code agree exactly" — inaccurate at the
  byte level (opt-in vs. always-on short filter; words vs. characters;
  undocumented leakage-permitting fallback) — flagged for correction
  once this milestone is implemented, not corrected yet.
- `peft==0.10.0` installed but unused by either shipped train script —
  unexplained, `UNKNOWN`, not guessed at.
- `vicuna_v1` conversation template's exact system-prompt string —
  `UNKNOWN` until LLaVA's own `conversation.py` is actually read (not
  vendored here).
- Module-layout reorganization (§7) diverges from the original Phase-1
  stub scaffold's file placement — must be reconciled explicitly when
  implementation begins, not silently overwritten.
- `reproduce_official_bug=True` experiments must never be run against
  real patient data without explicit, separate authorization — output
  rows self-flag (`reproduce_official_bug: true`) so they can never be
  mistaken for the project's default, safe output.

---

## 14. Open Questions (`UNKNOWN`)

- Exact `vicuna_v1` system-prompt string and role-separator tokens.
- Exact identity of the pretrained MM projector artifact referenced by
  `$PROJECTOR_PATH`.
- Why `peft==0.10.0` is installed but never invoked by the shipped
  training scripts.
- Current HuggingFace gating/license status of `lmsys/vicuna-7b-v1.5`
  and `liuhaotian/llava-v1.5-*` artifacts (only checkable via a real
  network reachability probe, reserved for Cell 29).
- Real GPU memory profile for both the full-fine-tune and LoRA MVP
  tracks — estimates only until an actual run is observed.

