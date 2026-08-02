"""Hard negative mining for the ANCE-style training stage.

Responsibility: reimplement the selection logic of the official
src/retriever/DPR/gen_hard_negatives.py -- for each query, take the
top-N embedding-similar candidates (FAISS in the official script; a
plain deterministic sort over caller-supplied embeddings here, since
FAISS/indexing is explicitly out of scope for this module), keep only
the candidates confirmed factually DISSIMILAR by *both* CheXbert and
RadGraph similarity (strict "<" thresholds, matching official code
exactly -- the opposite comparison direction from pair mining's
">="/">" positive thresholds), and select the num_top_neg *most*
dissimilar (lowest chexbert_sim + radgraph_sim) of those as hard
negatives.

This "hardest = most factually dissimilar among the embedding-closest"
selection looks inverted from the usual "hardest negative" intuition,
but it is intentional and paper-faithful: candidates are already
restricted to the embedding-top-N (so the retriever currently thinks
they're relevant), and among those, the ones that are in fact *most*
wrong are the ones most valuable for ANCE-style training -- exactly
the official code's own ranking rule, reproduced verbatim here.

Reuses chexbert_similarity/radgraph_similarity/combined_score from
src.baseline.pair_mining.similarity unchanged (identical scoring
functions; only the threshold direction and the identity of what is
being ranked differ from src.baseline.pair_mining.mining).

Embeddings are accepted as plain Sequence[float] (works for a Python
list/tuple, a numpy array, or a 1-D torch tensor via iteration) --
this module has no torch/numpy/FAISS dependency of its own.

Deliberate safety additions beyond the official script (which only
ever excludes the query's own corpus index by position):
  - a candidate already mined as a confirmed positive for this query
    (via the supplied Pair Mining output) is always excluded from
    becoming a negative;
  - a same-patient candidate is excluded by default
    (HardNegativeMiningConfig.exclude_same_patient=True) -- the
    official script has no patient-awareness at all;
  - a cross-dataset candidate is always rejected outright, mirroring
    PairMiner's own same-dataset corpus restriction.

Split isolation is NOT implemented here: corpus_records is assumed by
the caller to already be scoped to a single split (e.g. train-only),
exactly as PairMiner assumes of its own corpus_records -- this module
performs no split detection of its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Tuple

from src.baseline.pair_mining.mining import QueryKey
from src.baseline.pair_mining.similarity import chexbert_similarity, combined_score, radgraph_similarity
from src.common.exceptions import RetrievalDatasetError

_SIMILARITY_METRICS: Tuple[str, ...] = ("dot", "cosine")

# gen_hard_negatives.py's own argparse defaults (OFFICIAL_REPOSITORY),
# recorded verbatim, never silently substituted for the actual default
# below.
_OFFICIAL_REPOSITORY_DEFAULTS = {
    "top_n": 100,
    "num_top_neg": 2,
    "chex_threshold": 1.0,
    "radg_threshold": 0.4,
}


@dataclass(frozen=True)
class HardNegativeMiningConfig:
    config_version: str = "1.0"
    top_n: int = 100
    num_top_neg: int = 2
    similarity_metric: Literal["dot", "cosine"] = "dot"
    chex_threshold: float = 1.0
    radg_threshold: float = 0.4
    exclude_same_patient: bool = True

    def __post_init__(self) -> None:
        if not self.config_version:
            raise ValueError("config_version must be a non-empty string")
        if not isinstance(self.top_n, int) or isinstance(self.top_n, bool) or self.top_n < 1:
            raise ValueError(f"top_n must be a positive integer, got {self.top_n!r}")
        if (
            not isinstance(self.num_top_neg, int)
            or isinstance(self.num_top_neg, bool)
            or self.num_top_neg < 1
        ):
            raise ValueError(f"num_top_neg must be a positive integer, got {self.num_top_neg!r}")
        if self.similarity_metric not in _SIMILARITY_METRICS:
            raise ValueError(
                f"similarity_metric must be one of {_SIMILARITY_METRICS}, got {self.similarity_metric!r}"
            )
        if not (0.0 <= self.chex_threshold <= 1.0):
            raise ValueError(f"chex_threshold must be in [0.0, 1.0], got {self.chex_threshold!r}")
        if not (0.0 <= self.radg_threshold <= 1.0):
            raise ValueError(f"radg_threshold must be in [0.0, 1.0], got {self.radg_threshold!r}")
        if not isinstance(self.exclude_same_patient, bool):
            raise ValueError("exclude_same_patient must be a bool")

    def as_actual_used_dict(self) -> dict:
        return {
            "config_version": self.config_version,
            "top_n": self.top_n,
            "num_top_neg": self.num_top_neg,
            "similarity_metric": self.similarity_metric,
            "chex_threshold": self.chex_threshold,
            "radg_threshold": self.radg_threshold,
            "exclude_same_patient": self.exclude_same_patient,
        }


@dataclass(frozen=True)
class HardNegativeMiningResult:
    query_key: QueryKey
    negative_keys: List[QueryKey]
    scores: List[Dict[str, float]]  # parallel to negative_keys
    num_embedding_candidates_considered: int
    num_qualifying_candidates: int
    num_selected: int


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    denom = _norm(a) * _norm(b)
    if denom == 0.0:
        return 0.0
    return _dot(a, b) / denom


def _validate_embeddings(embeddings: Dict[QueryKey, Sequence[float]]) -> int:
    """Validates embeddings and returns the resolved (shared) dimension.

    Raises RetrievalDatasetError if embeddings is empty, if any two
    embeddings disagree in dimension, or if two distinct keys carry an
    identical embedding vector (a corrupted/duplicated-upstream-data
    signal, never silently treated as two legitimately distinct
    candidates).
    """
    if not embeddings:
        raise RetrievalDatasetError("embeddings must not be empty")

    resolved_dim = None
    seen_vectors: Dict[tuple, QueryKey] = {}
    for key, vector in embeddings.items():
        vector_tuple = tuple(float(v) for v in vector)
        if resolved_dim is None:
            resolved_dim = len(vector_tuple)
        elif len(vector_tuple) != resolved_dim:
            raise RetrievalDatasetError(
                f"Embedding dimension mismatch for key {key}: expected "
                f"{resolved_dim}, got {len(vector_tuple)}"
            )
        if vector_tuple in seen_vectors:
            raise RetrievalDatasetError(
                f"Duplicate embedding vector detected for keys "
                f"{seen_vectors[vector_tuple]} and {key}"
            )
        seen_vectors[vector_tuple] = key
    return resolved_dim


class HardNegativeMiner:
    """Mines num_top_neg hard negatives per query against a fixed corpus.

    corpus_records are plain dicts matching RadGraphAnnotator's JSONL
    schema (dataset, patient_id, study_id, entities, chexbert_labels_5,
    ...) -- the same shape src.baseline.retrieval.dataset.load_annotated_records
    produces. embeddings maps every corpus_records key to a fixed-length
    embedding vector (see module docstring). pair_mining_positives maps
    a query key to its already-mined positive keys (the shape
    src.baseline.retrieval.dataset.load_pair_mining_pairs produces) --
    used only to exclude confirmed positives from ever being selected
    as a negative.

    Construction validates:
      - embeddings is non-empty, every vector shares one resolved
        dimension, and no two distinct keys share an identical vector
        (see _validate_embeddings);
      - embeddings' key set exactly matches corpus_records' key set
        (key consistency) -- a record with no embedding, or an
        embedding with no record, both raise;
      - every positive key referenced in pair_mining_positives exists
        in corpus_records and shares its query's dataset.
    """

    def __init__(
        self,
        corpus_records: Dict[QueryKey, dict],
        embeddings: Dict[QueryKey, Sequence[float]],
        pair_mining_positives: Dict[QueryKey, List[QueryKey]],
        *,
        config: HardNegativeMiningConfig,
        corpus_scope: str = "train_only",
    ) -> None:
        self._embedding_dim = _validate_embeddings(embeddings)

        record_keys = set(corpus_records)
        embedding_keys = set(embeddings)
        if record_keys != embedding_keys:
            missing_embeddings = sorted(record_keys - embedding_keys)
            extra_embeddings = sorted(embedding_keys - record_keys)
            raise RetrievalDatasetError(
                "corpus_records and embeddings key sets do not match -- "
                f"missing embeddings for {missing_embeddings[:5]}"
                f"{'...' if len(missing_embeddings) > 5 else ''}, "
                f"unexpected embeddings for {extra_embeddings[:5]}"
                f"{'...' if len(extra_embeddings) > 5 else ''}"
            )

        for query_key, positive_keys in pair_mining_positives.items():
            for positive_key in positive_keys:
                if positive_key not in corpus_records:
                    raise RetrievalDatasetError(
                        f"Positive key {positive_key} referenced by query "
                        f"{query_key} has no matching corpus record"
                    )
                if positive_key[0] != query_key[0]:
                    raise RetrievalDatasetError(
                        f"Positive key {positive_key} violates the cross-dataset "
                        f"invariant (query dataset={query_key[0]!r})"
                    )

        self._corpus_records = corpus_records
        self._embeddings = embeddings
        self._pair_mining_positives = pair_mining_positives
        self._config = config
        self._corpus_scope = corpus_scope

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _embedding_similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        if self._config.similarity_metric == "cosine":
            return _cosine(a, b)
        return _dot(a, b)

    def mine_query(self, query_key: QueryKey) -> HardNegativeMiningResult:
        if query_key not in self._corpus_records:
            raise RetrievalDatasetError(f"Query key {query_key} has no matching corpus record")

        query_record = self._corpus_records[query_key]
        query_embedding = self._embeddings[query_key]

        candidate_keys = [
            key
            for key in self._corpus_records
            if key != query_key and self._corpus_records[key]["dataset"] == query_record["dataset"]
        ]
        if self._config.exclude_same_patient:
            candidate_keys = [key for key in candidate_keys if key[1] != query_key[1]]

        confirmed_positives = set(self._pair_mining_positives.get(query_key, []))
        candidate_keys = [key for key in candidate_keys if key not in confirmed_positives]

        num_embedding_candidates_considered = len(candidate_keys)

        # Deterministic descending-similarity ranking with a fully
        # deterministic tie-break -- never an implementation-defined
        # unstable ordering (matching PairMiner's own tie-breaking
        # discipline). Ordering is a function only of key identity and
        # score, so it is independent of self._corpus_records'/
        # self._embeddings' own (arbitrary) dict iteration order.
        scored = [
            (key, self._embedding_similarity(query_embedding, self._embeddings[key]))
            for key in candidate_keys
        ]
        scored.sort(key=lambda item: (-item[1], item[0][0], item[0][1], item[0][2]))

        top_n = scored[: self._config.top_n]

        qualifying: List[Tuple[QueryKey, float, float]] = []
        for candidate_key, _embedding_sim in top_n:
            candidate_record = self._corpus_records[candidate_key]
            chex_sim = chexbert_similarity(
                query_record["chexbert_labels_5"], candidate_record["chexbert_labels_5"]
            )
            if chex_sim >= self._config.chex_threshold:
                continue
            radg_sim = radgraph_similarity(query_record["entities"], candidate_record["entities"])
            if radg_sim >= self._config.radg_threshold:
                continue
            qualifying.append((candidate_key, chex_sim, radg_sim))

        num_qualifying_candidates = len(qualifying)

        # Hardest = lowest combined_score first (most factually
        # dissimilar among the embedding-top-N) -- see module
        # docstring. Deterministic tie-break identical in form to
        # PairMiner's.
        qualifying.sort(
            key=lambda item: (
                combined_score(item[1], item[2]),
                item[0][0],
                item[0][1],
                item[0][2],
            )
        )

        selected = qualifying[: self._config.num_top_neg]
        negative_keys = [item[0] for item in selected]

        if len(set(negative_keys)) != len(negative_keys):
            raise RetrievalDatasetError(
                f"Internal invariant violated: duplicate negative keys selected "
                f"for query {query_key}: {negative_keys}"
            )

        scores = [
            {
                "chexbert_similarity": item[1],
                "radgraph_similarity": item[2],
                "combined_score": combined_score(item[1], item[2]),
            }
            for item in selected
        ]

        return HardNegativeMiningResult(
            query_key=query_key,
            negative_keys=negative_keys,
            scores=scores,
            num_embedding_candidates_considered=num_embedding_candidates_considered,
            num_qualifying_candidates=num_qualifying_candidates,
            num_selected=len(negative_keys),
        )

    def mine_all(self, query_keys: Sequence[QueryKey]) -> Dict[QueryKey, List[QueryKey]]:
        """Convenience wrapper returning only negative_keys per query --
        the exact shape RetrieverTrainingDataset's hard_negatives_by_key
        expects. Does not write or resume from any file; this is an
        in-memory helper only.
        """
        return {query_key: self.mine_query(query_key).negative_keys for query_key in query_keys}
