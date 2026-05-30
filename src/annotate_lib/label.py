"""Cheapest-wins label assignment + binary fallback.

Ports the non-elif overwrite logic from ``classifier/preprocess/preprocess_utils.py::label_complexity``
upstream, where assignments happen in the order multi → one → zero, so
the *last* successful (= simplest) strategy wins.

Binary fallback table (data_plan.md §2):
  * NQ / Trivia / SQuAD → ``single_step``
  * HotpotQA / 2WikiMultiHopQA / MuSiQue → ``multi_step``
"""

from __future__ import annotations

from typing import Literal


Strategy = Literal["no_retrieval", "single_step", "multi_step"]
LabelSource = Literal["silver", "binary_fallback"]


_FALLBACK: dict[str, Strategy] = {
    "nq": "single_step",
    "trivia": "single_step",
    "squad": "single_step",
    "hotpotqa": "multi_step",
    "2wikimultihopqa": "multi_step",
    "musique": "multi_step",
}


def binary_fallback_for(dataset: str) -> Strategy:
    if dataset not in _FALLBACK:
        raise ValueError(f"No binary fallback for dataset {dataset!r}")
    return _FALLBACK[dataset]


def assign_label(
    *,
    attempts: dict[str, dict],
    source_dataset: str,
) -> tuple[Strategy, LabelSource]:
    """Return ``(oracle_label, label_source)``.

    Implements::

        if multi_step.ok:    label = "multi_step"
        if single_step.ok:   label = "single_step"   # overwrites multi
        if no_retrieval.ok:  label = "no_retrieval"  # overwrites single
        else:                label = binary_fallback(dataset)

    ``attempts`` need not contain all three keys (e.g. ``--strategies
    no_retrieval`` only). Missing strategies are treated as ``ok=False``.
    """
    def ok(name: str) -> bool:
        a = attempts.get(name)
        return bool(a and a.get("ok"))

    label: Strategy | None = None
    if ok("multi_step"):
        label = "multi_step"
    if ok("single_step"):
        label = "single_step"
    if ok("no_retrieval"):
        label = "no_retrieval"

    if label is not None:
        return label, "silver"
    return binary_fallback_for(source_dataset), "binary_fallback"
