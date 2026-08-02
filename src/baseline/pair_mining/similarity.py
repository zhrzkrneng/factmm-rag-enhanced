"""CheXbert and RadGraph similarity functions for factual pair mining.

Responsibility: pure, side-effect-free scoring functions matching the
official FactMM-RAG data/factual_mining/*/utils.py exactly --
chexbert_similarity (per-position agreement over the 5-class subset)
and radgraph_similarity (the paper's Eq. 1 F1-style entity+relation
overlap, treating facts as a set, never a multiset). combined_score is
this project's baseline ranking rule (chexbert_sim + radgraph_sim,
matching the official code's `tensor = chexbert + radgraph`) -- the
official-code-compatible baseline, not an innovation. It is only used
to rank candidates that have already passed both threshold masks (see
src.baseline.pair_mining.mining); it is never itself compared against
a threshold.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Sequence, Tuple, Union

# (tokens, label) for a relation-free entity, or (tokens, label, True)
# for one that participates in at least one relation -- only
# presence/absence of a relation is encoded, never its type or target.
_Fact = Union[Tuple[str, str], Tuple[str, str, bool]]


def chexbert_similarity(label_a: Sequence[int], label_b: Sequence[int]) -> float:
    """Fraction of matching positions between two equal-length label vectors.

    Matches the official utils.py::chexbert_similarity exactly: plain
    elementwise agreement (0s count as matches too), not an F1 score,
    not restricted to positive labels. Intended for the 5-class
    CheXbert subset (chexbert_labels_5) -- the official mining pipeline
    never touches the 14-class vector at all (label.py already reduces
    to 5 classes before mining sees the data).

    Raises:
        ValueError: label_a/label_b differ in length, or are empty.
    """
    if len(label_a) != len(label_b):
        raise ValueError(
            f"chexbert_similarity requires equal-length vectors, got "
            f"{len(label_a)} and {len(label_b)}"
        )
    if len(label_a) == 0:
        raise ValueError("chexbert_similarity requires non-empty label vectors")
    matches = sum(1 for a, b in zip(label_a, label_b) if a == b)
    return matches / len(label_a)


def _entity_fact_set(entities: Dict[str, dict]) -> FrozenSet[_Fact]:
    facts = set()
    for entity in entities.values():
        if entity["relations"]:
            facts.add((entity["tokens"], entity["label"], True))
        else:
            facts.add((entity["tokens"], entity["label"]))
    return frozenset(facts)


def radgraph_similarity(entities_a: Dict[str, dict], entities_b: Dict[str, dict]) -> float:
    """Symmetric F1-style overlap between two RadGraph entity sets (paper Eq. 1).

    Matches the official utils.py::exact_entity_token_if_rel_exists_reward
    exactly: each entity becomes (tokens, label) if it has no relations,
    or (tokens, label, True) if it has at least one -- relation TYPE and
    TARGET are discarded, only presence/absence is encoded. Facts are
    deduplicated as a set: a repeated identical fact counts once, not
    once per occurrence. precision = |a & b| / |a|, recall = |a & b| /
    |b|, F1 = harmonic mean of the two; an empty set on either side
    yields 0.0 for that term, and two empty sets yield an overall score
    of 0.0 (not 1.0 -- two reports with no extractable facts are not
    treated as "identical").

    The official code names its arguments "hypothesis"/"reference"
    asymmetrically, but the computation is symmetric in practice:
    precision and recall both key off the same intersection, so
    swapping entities_a/entities_b leaves the result unchanged. This
    function is written directly as the symmetric form.
    """
    set_a = _entity_fact_set(entities_a)
    set_b = _entity_fact_set(entities_b)
    intersection = len(set_a & set_b)
    precision = intersection / len(set_a) if set_a else 0.0
    recall = intersection / len(set_b) if set_b else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def combined_score(chexbert_sim: float, radgraph_sim: float) -> float:
    """combined_score = chexbert_sim + radgraph_sim.

    The official-code-compatible baseline ranking rule -- a plain,
    unweighted sum, matching the official code's
    `tensor = chexbert + radgraph` exactly (no averaging, no
    weighting). Domain: chexbert_sim in [0,1], radgraph_sim in [0,1]
    -> range [0,2]. Used only to rank/select among candidates that
    have already passed both threshold masks; never itself compared
    against a threshold. This is the baseline ranking rule being
    reproduced, not an innovation of this project's own.
    """
    return chexbert_sim + radgraph_sim
