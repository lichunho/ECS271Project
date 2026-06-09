"""Offline tests for the evaluation harness (src/eval_lib + scripts/evaluate).

No GPU/JDK/LLM/Drive. Mirrors the ``tmp_path`` JSONL style of
``tests/test_annotate_merge.py``.

Run with::

    .\\.venv\\Scripts\\python.exe -m pytest tests\\test_eval_lib.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval_lib import discover, grid as grid_mod, metrics, probe_sweep, sources


# --- fixtures --------------------------------------------------------------

def _attempt(strategy: str, em: int, *, latency=1.0, ptok=100, ctok=10, hops=0) -> dict:
    return {
        "strategy": strategy,
        "pred_raw": "", "pred_extracted": "",
        "em": em, "f1": float(em), "ok": bool(em),
        "latency_s": latency,
        "prompt_tokens_est": ptok, "completion_tokens_est": ctok,
        "context_doc_ids": [], "n_hops": hops, "error": "",
    }


def _row(qid, dataset, oracle, label_source, ems) -> dict:
    """``ems`` = (no_retrieval_em, single_step_em, multi_step_em)."""
    nr, ss, ms = ems
    return {
        "question_id": qid,
        "question_text": f"q-{qid}",
        "dataset": dataset,
        "oracle_label": oracle,
        "label_source": label_source,
        "labeller_model_id": "gemma4:26b",
        "attempts": {
            "no_retrieval": _attempt("no_retrieval", nr, hops=0),
            "single_step": _attempt("single_step", ss, hops=1),
            "multi_step": _attempt("multi_step", ms, hops=2),
        },
    }


# Five questions across two datasets. q3 = "good" demotion (single & multi both
# right), q4 = "bad" demotion (only multi right), q5 = binary_fallback all wrong.
GRID_ROWS = {
    "nq": [
        _row("q1", "nq", "no_retrieval", "silver", (1, 1, 0)),
        _row("q2", "nq", "single_step", "silver", (0, 1, 1)),
    ],
    "musique": [
        _row("q3", "musique", "multi_step", "silver", (0, 1, 1)),
        _row("q4", "musique", "multi_step", "silver", (0, 0, 1)),
        _row("q5", "musique", "multi_step", "binary_fallback", (0, 0, 0)),
    ],
}


def _write_grid(labeled_dir: Path, split="eval", rows=None):
    rows = rows or GRID_ROWS
    for ds, drows in rows.items():
        path = labeled_dir / split / f"{ds}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf8") as f:
            for r in drows:
                f.write(json.dumps(r) + "\n")


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _routes_row(qid, dataset, initial_route, ms=10.0):
    return {
        "question_id": qid, "dataset": dataset,
        "initial_route": initial_route, "classifier_inference_ms": ms,
    }


def _probed_row(qid, dataset, initial_route, min_prob, final_route, ms=10.0, probe_ms=50.0):
    return {
        "question_id": qid, "dataset": dataset,
        "initial_route": initial_route, "classifier_inference_ms": ms,
        "probe_first_sentence_min_prob": min_prob,
        "probe_latency_ms": probe_ms, "final_route": final_route,
    }


@pytest.fixture
def grid(tmp_path):
    _write_grid(tmp_path / "labeled")
    return grid_mod.load_grid(tmp_path / "labeled", "eval", ["nq", "musique"])


# --- 2. loader -------------------------------------------------------------

def test_loader_qids_and_attempts(grid):
    assert set(grid) == {"q1", "q2", "q3", "q4", "q5"}
    assert grid["q1"].oracle_label == "no_retrieval"
    assert grid["q1"].attempts["single_step"].em == 1
    assert grid["q5"].label_source == "binary_fallback"
    assert grid["q5"].is_silver is False


def test_loader_duplicate_qid_raises(tmp_path):
    dup = {"nq": [_row("dup", "nq", "no_retrieval", "silver", (1, 0, 0)),
                  _row("dup", "nq", "single_step", "silver", (0, 1, 0))]}
    _write_grid(tmp_path / "labeled", rows=dup)
    with pytest.raises(ValueError, match="duplicate"):
        grid_mod.load_grid(tmp_path / "labeled", "eval", ["nq"])


# --- 3. constant source ----------------------------------------------------

def test_constant_source_em_matches_strategy_mean(grid):
    ceil_em, ceil_f1 = metrics.oracle_ceiling(grid)
    src = sources.constant_source(grid, "single_step")
    res = metrics.evaluate_source(grid, src, ceil_em, ceil_f1)
    # single_step ems across q1..q5 = 1,1,1,0,0 -> 0.6
    assert res.em_mean == pytest.approx(0.6)
    assert res.f1_mean == pytest.approx(0.6)


# --- 4. oracle ceiling -----------------------------------------------------

def test_oracle_source_em_equals_ceiling(grid):
    ceil_em, ceil_f1 = metrics.oracle_ceiling(grid)
    # oracle ems = q1 no_retrieval(1), q2 single(1), q3 multi(1), q4 multi(1), q5 multi(0) -> 0.8
    assert ceil_em == pytest.approx(0.8)
    src = sources.oracle_source(grid)
    res = metrics.evaluate_source(grid, src, ceil_em, ceil_f1)
    assert res.em_mean == pytest.approx(ceil_em)
    # oracle is always correct on silver rows by construction
    assert res.regret["parity"] == pytest.approx(1.0)
    assert res.regret["regret"] == pytest.approx(0.0)


# --- 5. routing confusion / per-class / silver split -----------------------

def test_routing_confusion_and_silver_split(tmp_path, grid):
    rfile = tmp_path / "clf.routes.jsonl"
    _write_jsonl(rfile, [
        _routes_row("q1", "nq", "no_retrieval"),       # true no_retrieval -> correct
        _routes_row("q2", "nq", "multi_step"),          # true single_step -> wrong
        _routes_row("q3", "musique", "multi_step"),     # true multi_step  -> correct
        _routes_row("q4", "musique", "single_step"),    # true multi_step  -> wrong
        _routes_row("q5", "musique", "multi_step"),     # fallback, true multi_step
    ])
    ceil_em, ceil_f1 = metrics.oracle_ceiling(grid)
    src = sources.from_routes_file(rfile)
    res = metrics.evaluate_source(grid, src, ceil_em, ceil_f1)

    assert src.name == "clf"
    # all rows: diagonal = q1,q3,q5 -> 3/5
    assert res.routing_all["accuracy"] == pytest.approx(0.6)
    # silver only excludes q5: diagonal = q1,q3 -> 2/4
    assert res.routing_silver["accuracy"] == pytest.approx(0.5)
    conf = res.routing_all["confusion"]
    assert conf["no_retrieval"]["no_retrieval"] == 1
    assert conf["single_step"]["multi_step"] == 1
    assert conf["multi_step"]["multi_step"] == 2
    assert conf["multi_step"]["single_step"] == 1
    rec = res.routing_all["per_class_recall"]
    assert rec["no_retrieval"] == pytest.approx(1.0)
    assert rec["single_step"] == pytest.approx(0.0)
    assert rec["multi_step"] == pytest.approx(2 / 3)
    # EM of chosen routes: q1(1)+q2 multi(1)+q3 multi(1)+q4 single(0)+q5 multi(0) = 0.6
    assert res.em_mean == pytest.approx(0.6)


# --- 6. percentile helper --------------------------------------------------

def test_percentile_nearest_rank():
    vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert metrics._percentile(vals, 50) == 5.0
    assert metrics._percentile(vals, 90) == 9.0
    assert metrics._percentile([], 90) == 0.0


# --- 7. tau-sweep ----------------------------------------------------------

def test_tau_sweep_demotion_counts_and_never_demote(tmp_path, grid):
    pfile = tmp_path / "clf.probed.jsonl"
    _write_jsonl(pfile, [
        # non-multi rows: passthrough, never demoted
        _probed_row("q1", "nq", "no_retrieval", None, "no_retrieval"),
        _probed_row("q2", "nq", "single_step", None, "single_step"),
        # multi rows with distinct min_probs
        _probed_row("q3", "musique", "multi_step", 0.8, "single_step"),  # good (single em=1)
        _probed_row("q4", "musique", "multi_step", 0.3, "multi_step"),   # bad if demoted
        _probed_row("q5", "musique", "multi_step", 0.95, "single_step"),
    ])
    sw = probe_sweep.sweep_probe(grid, pfile, [0.2, 0.5, 0.9])
    pts = {p["tau"]: p for p in sw.points}
    assert sw.name == "clf"

    # tau=0.2: q3,q4,q5 all demote (min_prob>=0.2)
    assert pts[0.2]["n_demoted"] == 3
    assert pts[0.2]["good_demotions"] == 1   # q3 single em=1
    assert pts[0.2]["bad_demotions"] == 1    # q4 single em=0 but multi em=1 (q5 all-0 -> neither)

    # tau=0.5: q3 (0.8) and q5 (0.95) demote, q4 (0.3) stays
    assert pts[0.5]["n_demoted"] == 2
    assert pts[0.5]["bad_demotions"] == 0

    # tau=0.9: only q5 (0.95) demotes
    assert pts[0.9]["n_demoted"] == 1


def test_tau_high_equals_never_demote(tmp_path, grid):
    pfile = tmp_path / "clf.probed.jsonl"
    rows = [
        _probed_row("q3", "musique", "multi_step", 0.8, "single_step"),
        _probed_row("q4", "musique", "multi_step", 0.3, "multi_step"),
    ]
    _write_jsonl(pfile, rows)
    sw = probe_sweep.sweep_probe(grid, pfile, [2.0])  # tau above any prob
    assert sw.points[0]["n_demoted"] == 0
    # equals the routes-file EM (route == initial multi for both; multi em=1,1)
    assert sw.points[0]["em_mean"] == pytest.approx(1.0)


def test_min_prob_none_never_demoted():
    row = {"initial_route": "multi_step", "probe_first_sentence_min_prob": None}
    assert sources.route_at_tau(row, 0.0) == "multi_step"
    assert sources.route_at_tau(row, 1.0) == "multi_step"


# --- 8. coverage -----------------------------------------------------------

def test_coverage_counts_missing_both_ways(tmp_path, grid):
    rfile = tmp_path / "clf.routes.jsonl"
    _write_jsonl(rfile, [
        _routes_row("q1", "nq", "no_retrieval"),
        _routes_row("qX", "nq", "single_step"),  # not in grid
    ])
    ceil_em, ceil_f1 = metrics.oracle_ceiling(grid)
    src = sources.from_routes_file(rfile)
    res = metrics.evaluate_source(grid, src, ceil_em, ceil_f1)
    assert res.coverage["matched"] == 1           # only q1
    assert res.coverage["missing_in_grid"] == 1   # qX
    assert res.coverage["missing_in_source"] == 4  # q2..q5
    assert res.coverage["grid_total"] == 5


# --- 9. discovery ----------------------------------------------------------

def test_discovery_lists_routes_probed_and_baselines(tmp_path, grid):
    inp = tmp_path / "routes"
    _write_jsonl(inp / "roberta.routes.jsonl", [_routes_row("q1", "nq", "no_retrieval")])
    _write_jsonl(inp / "roberta.probed.jsonl", [_probed_row("q3", "musique", "multi_step", 0.8, "single_step")])
    _write_jsonl(inp / "t5.routes.jsonl", [_routes_row("q2", "nq", "single_step")])

    srcs = discover.discover_sources(inp, grid, grid, include_baselines=True)
    names = [s.name for s in srcs]
    assert "roberta" in names
    assert "roberta (+probe)" in names
    assert "t5" in names
    # six synthetic baselines appended
    for base in ["always-no_retrieval", "always-single_step", "always-multi_step", "oracle", "random"]:
        assert base in names
    assert any(n.startswith("majority(") for n in names)

    srcs_nb = discover.discover_sources(inp, grid, {}, include_baselines=False)
    assert len(srcs_nb) == 3


# --- 10. real-data smoke (skipped if labelled data absent) -----------------

def test_real_data_oracle_ceiling_smoke():
    from src import config
    eval_dir = config.LABELED_DIR / "eval"
    if not (eval_dir / "musique.jsonl").exists():
        pytest.skip("labelled eval data not present")
    g = grid_mod.load_grid(config.LABELED_DIR, "eval")
    ceil_em, _ = metrics.oracle_ceiling(g)
    assert 0.30 < ceil_em < 0.55  # ~0.417 per the plan
    src = sources.oracle_source(g)
    res = metrics.evaluate_source(g, src, ceil_em, 0.0)
    assert res.em_mean == pytest.approx(ceil_em)
