"""Route sources: anything that maps a question_id to a chosen route + overhead.

A ``RouteSource`` is a named ``{qid: RouteDecision}`` map. Real classifiers come
from ``*.routes.jsonl`` (key ``initial_route``) and ``*.probed.jsonl`` (key
``final_route``, with probe latency). Synthetic baselines (constant / oracle /
majority / random) are derived straight from the grid.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .grid import ROUTES


@dataclass(frozen=True)
class RouteDecision:
    route: str
    classifier_ms: float = 0.0
    probe_ms: float = 0.0
    probe_tokens: int = 0  # 0 in v1 (probe token cost not yet recorded)


@dataclass
class RouteSource:
    name: str
    decisions: dict  # {qid: RouteDecision}
    kind: str = "classifier"  # "classifier" | "probe" | "baseline"

    def __len__(self) -> int:
        return len(self.decisions)


def _read_jsonl(path: Path):
    with open(path, "r", encoding="utf8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _strip_suffix(path: Path, suffix: str) -> str:
    """``foo.routes.jsonl`` -> ``foo`` (Path.stem only drops ``.jsonl``)."""
    stem = path.stem  # drops .jsonl
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem


# --- real classifiers ------------------------------------------------------

def from_routes_file(path: Path) -> RouteSource:
    """Read a ``*.routes.jsonl`` file; chosen route is ``initial_route``."""
    path = Path(path)
    decisions = {}
    for row in _read_jsonl(path):
        qid = str(row["question_id"])
        decisions[qid] = RouteDecision(
            route=row["initial_route"],
            classifier_ms=float(row.get("classifier_inference_ms", 0.0)),
        )
    return RouteSource(name=_strip_suffix(path, ".routes"), decisions=decisions)


def from_probed_file(path: Path) -> RouteSource:
    """Read a ``*.probed.jsonl`` file at its baked-in threshold; chosen route is
    ``final_route`` and probe latency is added as overhead."""
    path = Path(path)
    decisions = {}
    for row in _read_jsonl(path):
        qid = str(row["question_id"])
        decisions[qid] = RouteDecision(
            route=row["final_route"],
            classifier_ms=float(row.get("classifier_inference_ms", 0.0)),
            probe_ms=float(row.get("probe_latency_ms", 0.0) or 0.0),
        )
    return RouteSource(
        name=_strip_suffix(path, ".probed") + " (+probe)",
        decisions=decisions,
        kind="probe",
    )


def route_at_tau(row: dict, tau: float) -> str:
    """Re-derive the post-probe route for a probed row at threshold ``tau``.

    Probe semantics (from ``probe_multi_routes.probe_one``): a multi_step
    prediction is demoted to single_step iff *every* first-sentence token
    probability >= threshold, i.e. ``min_prob >= tau``. Non-multi rows and rows
    without usable logprobs (``min_prob is None``) are never demoted.
    """
    if row.get("initial_route") != "multi_step":
        return row.get("initial_route")
    min_prob = row.get("probe_first_sentence_min_prob")
    if min_prob is None:
        return "multi_step"
    return "single_step" if float(min_prob) >= tau else "multi_step"


def from_probed_file_at_tau(path: Path, tau: float) -> RouteSource:
    """Same as ``from_probed_file`` but re-derives the route at an arbitrary
    ``tau`` from the stored first-sentence min probability."""
    path = Path(path)
    decisions = {}
    for row in _read_jsonl(path):
        qid = str(row["question_id"])
        decisions[qid] = RouteDecision(
            route=route_at_tau(row, tau),
            classifier_ms=float(row.get("classifier_inference_ms", 0.0)),
            probe_ms=float(row.get("probe_latency_ms", 0.0) or 0.0),
        )
    return RouteSource(
        name=f"{_strip_suffix(path, '.probed')} (+probe@{tau:g})",
        decisions=decisions,
        kind="probe",
    )


# --- synthetic baselines ---------------------------------------------------

def constant_source(grid: dict, route: str) -> RouteSource:
    """Always pick ``route`` (zero overhead)."""
    decisions = {qid: RouteDecision(route=route) for qid in grid}
    return RouteSource(name=f"always-{route}", decisions=decisions, kind="baseline")


def oracle_source(grid: dict) -> RouteSource:
    """Always pick each row's ``oracle_label`` (the cheapest correct strategy)."""
    decisions = {
        qid: RouteDecision(route=gr.oracle_label) for qid, gr in grid.items()
    }
    return RouteSource(name="oracle", decisions=decisions, kind="baseline")


def majority_source(grid: dict, fit_grid: dict) -> RouteSource:
    """Pick the single most common ``oracle_label`` in ``fit_grid`` for every
    question (a degenerate classifier baseline). Ties broken by ``ROUTES`` order."""
    counts = {r: 0 for r in ROUTES}
    for gr in fit_grid.values():
        counts[gr.oracle_label] = counts.get(gr.oracle_label, 0) + 1
    majority = max(ROUTES, key=lambda r: counts.get(r, 0))
    src = constant_source(grid, majority)
    src.name = f"majority({majority})"
    return src


def random_source(grid: dict, seed: int = 13370) -> RouteSource:
    """Uniform-random route per question, deterministic under ``seed``."""
    rng = random.Random(seed)
    decisions = {
        qid: RouteDecision(route=rng.choice(ROUTES)) for qid in sorted(grid)
    }
    return RouteSource(name="random", decisions=decisions, kind="baseline")


def synthetic_baselines(grid: dict, fit_grid: dict) -> list:
    """The six reference points that bracket every real classifier."""
    return [
        constant_source(grid, "no_retrieval"),
        constant_source(grid, "single_step"),
        constant_source(grid, "multi_step"),
        oracle_source(grid),
        majority_source(grid, fit_grid),
        random_source(grid),
    ]
