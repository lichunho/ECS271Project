"""SQuAD ``normalize_answer`` and EM / F1 metrics.

Ports `evaluate.py::normalize_answer` and `metrics/squad_answer_em_f1.py`
from starsuzi/Adaptive-RAG verbatim. EM is the correctness gate; F1 is
logged but never gates a label assignment (per data_plan.md §2).
"""

from __future__ import annotations

import collections
import re
import string
from typing import Iterable


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = frozenset(string.punctuation)


def normalize_answer(s: str) -> str:
    """Lower, strip punctuation, drop articles, collapse whitespace.

    Exactly the function used by ``evaluate.py`` upstream::

        return white_space_fix(remove_articles(remove_punc(lower(s))))
    """
    if s is None:
        return ""
    # lower
    s = s.lower()
    # remove punctuation (single pass over chars)
    s = "".join(ch for ch in s if ch not in _PUNCT)
    # remove articles
    s = _ARTICLES_RE.sub(" ", s)
    # collapse whitespace
    s = " ".join(s.split())
    return s


def _tokens(s: str) -> list[str]:
    """Tokens after ``normalize_answer`` — matches upstream ``get_tokens``."""
    n = normalize_answer(s)
    if not n:
        return []
    return n.split()


def em_single(pred: str, gold: str) -> int:
    """Exact-match of one prediction vs one gold (0/1)."""
    return int(normalize_answer(pred) == normalize_answer(gold))


def f1_single(pred: str, gold: str) -> float:
    """Token-overlap F1 of one prediction vs one gold (0..1).

    Mirrors ``metrics/squad_answer_em_f1.py::compute_f1``.
    """
    pred_toks = _tokens(pred)
    gold_toks = _tokens(gold)
    if not gold_toks or not pred_toks:
        # Upstream returns 1 if both empty, else 0.
        return float(gold_toks == pred_toks)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def em(pred: str, golds: Iterable[str]) -> int:
    """Max EM over alias list."""
    gs = [g for g in golds if g is not None]
    if not gs:
        return 0
    return max(em_single(pred, g) for g in gs)


def f1(pred: str, golds: Iterable[str]) -> float:
    """Max F1 over alias list."""
    gs = [g for g in golds if g is not None]
    if not gs:
        return 0.0
    return max(f1_single(pred, g) for g in gs)
