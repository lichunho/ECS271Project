"""Offline tau-sweep for a single ``*.probed.jsonl`` file.

The probe demotes a multi_step prediction to single_step iff the stored
first-sentence ``min_prob >= tau``. Because every strategy's metrics are in the
grid, we can re-derive the route at any tau and read off EM/F1/tokens/latency
without re-running the probe. We also count "good" demotions (single_step would
have been correct anyway) vs "bad" ones (multi_step was right, single isn't).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .sources import route_at_tau


@dataclass
class ProbeSweep:
    name: str            # classifier stem (no "(+probe)" suffix)
    points: list         # [{tau, em_mean, ..., good_demotions, bad_demotions}]


def _read_rows(path: Path) -> list:
    rows = []
    with open(path, "r", encoding="utf8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sweep_probe(grid: dict, probed_path: Path, taus) -> ProbeSweep:
    probed_path = Path(probed_path)
    rows = _read_rows(probed_path)
    name = probed_path.stem
    if name.endswith(".probed"):
        name = name[: -len(".probed")]

    points = []
    for tau in taus:
        ems, f1s, tokens, latencies = [], [], [], []
        n_demoted = good = bad = 0
        matched = 0
        for row in rows:
            qid = str(row["question_id"])
            gr = grid.get(qid)
            if gr is None:
                continue
            matched += 1
            chosen = route_at_tau(row, tau)
            a = gr.attempts[chosen]
            ems.append(a.em)
            f1s.append(a.f1)
            tokens.append(a.total_tokens)
            latencies.append(
                float(row.get("classifier_inference_ms", 0.0))
                + float(row.get("probe_latency_ms", 0.0) or 0.0)
                + a.latency_s * 1000.0
            )
            # demotion bookkeeping (only multi->single transitions count)
            if row.get("initial_route") == "multi_step" and chosen == "single_step":
                n_demoted += 1
                single_em = gr.attempts["single_step"].em
                multi_em = gr.attempts["multi_step"].em
                if single_em == 1:
                    good += 1
                elif multi_em == 1:
                    bad += 1
        points.append({
            "tau": tau,
            "n_matched": matched,
            "em_mean": (sum(ems) / len(ems)) if ems else 0.0,
            "f1_mean": (sum(f1s) / len(f1s)) if f1s else 0.0,
            "token_mean": (sum(tokens) / len(tokens)) if tokens else 0.0,
            "latency_mean_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
            "n_demoted": n_demoted,
            "good_demotions": good,
            "bad_demotions": bad,
        })
    return ProbeSweep(name=name, points=points)
