"""Render EvalResults + probe sweeps into ``results.json`` (full blob),
``leaderboard.md`` (paste-ready) and ``sweeps.md`` (per-classifier tau curves).
"""

from __future__ import annotations

import json
from pathlib import Path


def _pct(x) -> str:
    return "—" if x is None else f"{x * 100:.1f}"


def _num(x, places: int = 0) -> str:
    if x is None:
        return "—"
    return f"{x:,.{places}f}"


def _leaderboard_md(results, ceiling_em, ceiling_f1, split: str) -> str:
    # EM desc, then cheaper (token_mean asc) wins ties.
    ordered = sorted(results, key=lambda r: (-r.em_mean, r.token_mean))
    lines = [
        f"# Routing leaderboard — `{split}` split",
        "",
        f"Oracle ceiling: **{ceiling_em * 100:.1f} EM** / {ceiling_f1 * 100:.1f} F1 "
        "(best achievable by routing alone; the `oracle` row reaches it).",
        "",
        "| Source | EM | F1 | Tokens (mean) | Latency ms (mean) | "
        "Route acc (all) | Route acc (silver) | Regret | Cov |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ordered:
        cov = r.coverage
        cov_str = f"{cov['matched']}/{cov['grid_total']}"
        lines.append(
            f"| {r.name} | {_pct(r.em_mean)} | {_pct(r.f1_mean)} | "
            f"{_num(r.token_mean)} | {_num(r.latency_mean_ms)} | "
            f"{_pct(r.routing_all['accuracy'])} | {_pct(r.routing_silver['accuracy'])} | "
            f"{_pct(r.regret['regret'])} | {cov_str} |"
        )
    lines += [
        "",
        "_EM/F1/Route-acc/Regret in %. Routing accuracy is reported both over "
        "all rows and over silver rows only (binary_fallback rows can't change "
        "correctness). Regret = 1 − silver parity. Cov = matched / grid total._",
        "",
    ]
    return "\n".join(lines)


def _sweeps_md(sweeps, split: str) -> str:
    if not sweeps:
        return f"# Probe τ-sweeps — `{split}` split\n\n_No probed files found._\n"
    lines = [f"# Probe τ-sweeps — `{split}` split", ""]
    for sw in sweeps:
        lines += [
            f"## {sw.name}",
            "",
            "| τ | EM | F1 | Tokens (mean) | Latency ms (mean) | Demoted | Good | Bad |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for p in sw.points:
            lines.append(
                f"| {p['tau']:g} | {_pct(p['em_mean'])} | {_pct(p['f1_mean'])} | "
                f"{_num(p['token_mean'])} | {_num(p['latency_mean_ms'])} | "
                f"{p['n_demoted']} | {p['good_demotions']} | {p['bad_demotions']} |"
            )
        lines += [
            "",
            "_Good = demotion to single_step that stays correct; "
            "Bad = demotion that drops a multi_step that was right._",
            "",
        ]
    return "\n".join(lines)


def write_reports(results, sweeps, ceiling_em, ceiling_f1, output_dir: Path,
                  split: str) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    blob = {
        "split": split,
        "oracle_ceiling": {"em": ceiling_em, "f1": ceiling_f1},
        "results": [r.to_json() for r in results],
        "sweeps": [{"name": s.name, "points": s.points} for s in sweeps],
    }
    paths = {
        "results": output_dir / "results.json",
        "leaderboard": output_dir / "leaderboard.md",
        "sweeps": output_dir / "sweeps.md",
    }
    paths["results"].write_text(json.dumps(blob, indent=2), encoding="utf8")
    paths["leaderboard"].write_text(
        _leaderboard_md(results, ceiling_em, ceiling_f1, split), encoding="utf8"
    )
    paths["sweeps"].write_text(_sweeps_md(sweeps, split), encoding="utf8")
    return paths
