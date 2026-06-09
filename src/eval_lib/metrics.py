"""Join a RouteSource against the grid and aggregate the slide's three metric
groups (Answer Accuracy, Efficiency, Routing Accuracy) plus regret/coverage.

Everything is computed over the *intersection* of the source's decided qids and
the grid, so a partial route file never raises KeyError; coverage is reported
separately. EM/F1 are read from the grid, never recomputed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .grid import ROUTES


def _percentile(values, pct: float) -> float:
    """Nearest-rank percentile. ``pct`` in [0, 100]. Empty -> 0.0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return float(ordered[min(rank, len(ordered)) - 1])


def _dist(values) -> dict:
    return {
        "mean": (sum(values) / len(values)) if values else 0.0,
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
    }


def oracle_ceiling(grid: dict):
    """(em, f1) the oracle router would achieve: mean of each row's
    ``attempts[oracle_label]`` EM / F1. ~0.417 EM on the eval split."""
    if not grid:
        return 0.0, 0.0
    ems, f1s = [], []
    for gr in grid.values():
        a = gr.attempts[gr.oracle_label]
        ems.append(a.em)
        f1s.append(a.f1)
    return sum(ems) / len(ems), sum(f1s) / len(f1s)


def _empty_confusion() -> dict:
    return {t: {p: 0 for p in ROUTES} for t in ROUTES}


def _routing_block(confusion: dict) -> dict:
    total = sum(confusion[t][p] for t in ROUTES for p in ROUTES)
    correct = sum(confusion[t][t] for t in ROUTES)
    per_class_recall = {}
    for t in ROUTES:
        row_total = sum(confusion[t][p] for p in ROUTES)
        per_class_recall[t] = (confusion[t][t] / row_total) if row_total else None
    return {
        "n": total,
        "accuracy": (correct / total) if total else 0.0,
        "confusion": confusion,  # confusion[true_oracle][predicted_route]
        "per_class_recall": per_class_recall,
    }


@dataclass
class EvalResult:
    name: str
    kind: str
    coverage: dict
    accuracy: dict          # overall em/f1 + ceiling + per_dataset + per_oracle_class
    efficiency: dict        # tokens / latency_ms distributions
    routing_all: dict
    routing_silver: dict
    regret: dict

    # flat headline fields for leaderboard sorting/printing
    em_mean: float = 0.0
    f1_mean: float = 0.0
    token_mean: float = 0.0
    latency_mean_ms: float = 0.0

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "coverage": self.coverage,
            "accuracy": self.accuracy,
            "efficiency": self.efficiency,
            "routing_all": self.routing_all,
            "routing_silver": self.routing_silver,
            "regret": self.regret,
        }


def evaluate_source(grid: dict, source, ceiling_em: float, ceiling_f1: float) -> EvalResult:
    grid_ids = set(grid)
    src_ids = set(source.decisions)
    matched = sorted(grid_ids & src_ids)

    ems, f1s, tokens, latencies = [], [], [], []
    by_dataset: dict = {}
    by_class: dict = {}
    conf_all = _empty_confusion()
    conf_silver = _empty_confusion()
    silver_parity = []  # 1 if chosen attempt em==1 on a silver row

    for qid in matched:
        gr = grid[qid]
        chosen = source.decisions[qid].route
        a = gr.attempts[chosen]
        dec = source.decisions[qid]

        ems.append(a.em)
        f1s.append(a.f1)
        tokens.append(a.total_tokens + dec.probe_tokens)
        latencies.append(dec.classifier_ms + dec.probe_ms + a.latency_s * 1000.0)

        by_dataset.setdefault(gr.dataset, {"em": [], "f1": []})
        by_dataset[gr.dataset]["em"].append(a.em)
        by_dataset[gr.dataset]["f1"].append(a.f1)

        by_class.setdefault(gr.oracle_label, {"em": [], "f1": []})
        by_class[gr.oracle_label]["em"].append(a.em)
        by_class[gr.oracle_label]["f1"].append(a.f1)

        conf_all[gr.oracle_label][chosen] += 1
        if gr.is_silver:
            conf_silver[gr.oracle_label][chosen] += 1
            silver_parity.append(1 if a.em == 1 else 0)

    n = len(matched)
    em_mean = (sum(ems) / n) if n else 0.0
    f1_mean = (sum(f1s) / n) if n else 0.0

    accuracy = {
        "em_mean": em_mean,
        "f1_mean": f1_mean,
        "ceiling_em": ceiling_em,
        "ceiling_f1": ceiling_f1,
        "per_dataset": {
            ds: {
                "em_mean": sum(v["em"]) / len(v["em"]),
                "f1_mean": sum(v["f1"]) / len(v["f1"]),
                "n": len(v["em"]),
            }
            for ds, v in sorted(by_dataset.items())
        },
        "per_oracle_class": {
            c: {
                "em_mean": sum(v["em"]) / len(v["em"]),
                "f1_mean": sum(v["f1"]) / len(v["f1"]),
                "n": len(v["em"]),
            }
            for c, v in sorted(by_class.items())
        },
    }

    efficiency = {
        "tokens": _dist(tokens),
        "latency_ms": _dist(latencies),
    }

    parity = (sum(silver_parity) / len(silver_parity)) if silver_parity else None
    regret = {
        "n_silver": len(silver_parity),
        "parity": parity,
        "regret": (1.0 - parity) if parity is not None else None,
    }

    coverage = {
        "matched": n,
        "missing_in_grid": len(src_ids - grid_ids),
        "missing_in_source": len(grid_ids - src_ids),
        "grid_total": len(grid_ids),
    }

    return EvalResult(
        name=source.name,
        kind=source.kind,
        coverage=coverage,
        accuracy=accuracy,
        efficiency=efficiency,
        routing_all=_routing_block(conf_all),
        routing_silver=_routing_block(conf_silver),
        regret=regret,
        em_mean=em_mean,
        f1_mean=f1_mean,
        token_mean=efficiency["tokens"]["mean"],
        latency_mean_ms=efficiency["latency_ms"]["mean"],
    )
