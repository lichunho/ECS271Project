"""Adaptive-RAG oracle-labelling orchestrator (data_plan.md §2).

For every row in ``data/{eval_500,train_500}/{dataset}.jsonl`` we run up
to three answerer strategies — no_retrieval, single_step, multi_step —
under EM correctness gate, then assign the cheapest-wins oracle label
(with a per-dataset binary fallback when all three fail).

Outputs:

    data/labeled/eval/{dataset}.jsonl              # row + oracle_label + attempts
    data/labeled/train/{dataset}.jsonl             # ditto
    data/labeled/classifier_train.jsonl            # train merged across datasets
    data/labeled/classifier_valid.jsonl            # eval merged across datasets
    data/labeled/qc_report.json                    # per-(dataset,split) histograms

State:

    data/.annotate_state.json                      # which (dataset,split,strategy) are done
    data/.llm_cache.sqlite                         # per-call cache (WAL)

Examples (PowerShell):

    # Default: six datasets × both splits × three strategies
    .\\.venv\\Scripts\\python.exe scripts\\annotate.py

    # Smoke-test the no_retrieval path on 5 MuSiQue questions
    .\\.venv\\Scripts\\python.exe scripts\\annotate.py `
        --only-datasets musique --split eval --strategies no_retrieval --max-questions 5

    # Validate retrieval path (needs JDK 21 + Pyserini indices)
    .\\.venv\\Scripts\\python.exe scripts\\annotate.py `
        --only-datasets musique --split eval --strategies single_step --max-questions 2

    # Re-run everything from scratch (also clears state, not the SQLite cache).
    .\\.venv\\Scripts\\python.exe scripts\\annotate.py --force
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional

# Make ``src.*`` importable when running this script directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.annotate_lib import label as label_mod  # noqa: E402
from src.annotate_lib import prompts as prompts_mod  # noqa: E402
from src.annotate_lib import strategies as strat_mod  # noqa: E402
from src.annotate_lib.cache import AttemptCache, question_hash  # noqa: E402
from src.annotate_lib.llm_adapter import get_async_client  # noqa: E402
from src.annotate_lib.prompts import (  # noqa: E402
    ALL_DATASETS, COT_DATASETS, PROMPT_SET_ID,
)
from src.annotate_lib.strategies import Attempt  # noqa: E402
from src.config import LLM_MODEL  # noqa: E402


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("annotate")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = _REPO_ROOT / "data"
LABELED_DIR = DEFAULT_DATA_DIR / "labeled"
STATE_FILE = DEFAULT_DATA_DIR / ".annotate_state.json"
CACHE_FILE = DEFAULT_DATA_DIR / ".llm_cache.sqlite"

ALL_STRATEGIES = ("no_retrieval", "single_step", "multi_step")
SPLITS = ("eval", "train")
SPLIT_INPUT_DIR = {"eval": "eval_500", "train": "train_500"}


# ---------------------------------------------------------------------------
# State file
# ---------------------------------------------------------------------------


class State:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = {"completed": []}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf8"))
            except Exception as e:
                log.warning("Could not parse %s (%s); starting fresh.", path, e)

    def is_done(self, key: str) -> bool:
        return key in self.data.get("completed", [])

    def mark_done(self, key: str) -> None:
        if key not in self.data["completed"]:
            self.data["completed"].append(key)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf8")

    def clear(self) -> None:
        self.data = {"completed": []}
        if self.path.exists():
            self.path.unlink()


# ---------------------------------------------------------------------------
# Row I/O
# ---------------------------------------------------------------------------


def read_jsonl(p: Path) -> list[dict]:
    rows: list[dict] = []
    with open(p, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dataset_of(row: dict, fallback: str) -> str:
    """Some processed rows don't carry a top-level 'dataset' field — infer it."""
    ds = row.get("dataset")
    if ds:
        return ds
    return fallback


def collect_split_rows(
    split: str,
    labeled_dir: Path,
    per_split: dict[str, dict[str, list[dict]]] | None = None,
    datasets: Iterable[str] = ALL_DATASETS,
) -> list[dict]:
    """Union of all per-shard rows for ``split`` across ``datasets``.

    For each dataset we prefer freshly-labelled rows from
    ``per_split[split][ds]`` when present; otherwise we read the on-disk
    shard ``labeled_dir/split/{ds}.jsonl`` if it exists. Datasets with no
    fresh rows and no on-disk shard are skipped.

    This makes the merged classifier file complete regardless of the current
    run's ``--only-datasets`` scope: subset runs still pick up the other
    datasets' shards from disk (they were written by ``write_labeled_jsonl``
    before the merge). ``datasets`` defaults to :data:`ALL_DATASETS`.
    """
    fresh = (per_split or {}).get(split, {})
    rows: list[dict] = []
    for ds in datasets:
        if ds in fresh:
            rows.extend(fresh[ds])
            continue
        shard = labeled_dir / split / f"{ds}.jsonl"
        if shard.exists():
            rows.extend(read_jsonl(shard))
    return rows


def collect_full_per_split(
    splits: Iterable[str],
    labeled_dir: Path,
    per_split: dict[str, dict[str, list[dict]]] | None = None,
    datasets: Iterable[str] = ALL_DATASETS,
) -> dict[str, dict[str, list[dict]]]:
    """Per-split → per-dataset mapping covering ALL on-disk shards.

    For each split in ``splits`` and each dataset in ``datasets``, prefer the
    freshly-labelled rows in ``per_split`` (overlay), else load the on-disk
    shard. Datasets with no data are omitted. Used to feed both the merge and
    :func:`build_qc_report` so neither is scoped to the current run's subset.
    """
    fresh = per_split or {}
    out: dict[str, dict[str, list[dict]]] = {}
    for split in splits:
        split_fresh = fresh.get(split, {})
        out[split] = {}
        for ds in datasets:
            if ds in split_fresh:
                out[split][ds] = split_fresh[ds]
                continue
            shard = labeled_dir / split / f"{ds}.jsonl"
            if shard.exists():
                out[split][ds] = read_jsonl(shard)
    return out


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(_REPO_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf8").strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Per-question pipeline
# ---------------------------------------------------------------------------


async def _run_strategy(
    name: str,
    *,
    question: str,
    dataset: str,
    gold_aliases: list[str],
    model: str,
    bm25_k: int,
    ircot_max_iters: int,
    ircot_k_per_step: int,
    client,
    retriever,
) -> Attempt:
    if name == "no_retrieval":
        return await strat_mod.no_retrieval(
            question, dataset, gold_aliases=gold_aliases,
            model=model, client=client,
        )
    if name == "single_step":
        return await strat_mod.single_step(
            question, dataset, gold_aliases=gold_aliases,
            model=model, k=bm25_k, client=client, retriever=retriever,
        )
    if name == "multi_step":
        return await strat_mod.multi_step(
            question, dataset, gold_aliases=gold_aliases,
            model=model, max_iters=ircot_max_iters,
            k_per_step=ircot_k_per_step,
            client=client, retriever=retriever,
        )
    raise ValueError(f"Unknown strategy: {name}")


async def annotate_one_row(
    row: dict,
    *,
    source_dataset: str,
    strategies: tuple[str, ...],
    model: str,
    bm25_k: int,
    ircot_max_iters: int,
    ircot_k_per_step: int,
    cache: AttemptCache,
    client,
    retriever,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Return the labelled-row dict (Adaptive-RAG schema + oracle fields).

    Strategies passed via ``strategies`` are always (re-)evaluated, hitting
    the cache when possible and running the LLM otherwise. After that loop,
    we *opportunistically* fill ``attempts`` with any cached results for the
    other canonical strategies in :data:`ALL_STRATEGIES`. This keeps partial
    re-runs (e.g. ``--strategies single_step,multi_step`` over a corpus where
    ``no_retrieval`` was cached on a prior run) idempotent in two ways:

    * The emitted row carries the full ``attempts`` dict, so
      ``write_labeled_jsonl`` doesn't lose previously-recorded attempts when
      the file is regenerated.
    * ``label_mod.assign_label`` sees every cached strategy and applies
      cheapest-wins correctly — without this, a question whose cached
      ``no_retrieval`` would have been the cheapest pass gets mis-labelled
      ``single_step`` (or ``binary_fallback``).
    """
    question = row["question_text"]
    qh = question_hash(question)
    gold_aliases = strat_mod._gold_aliases(row)

    attempts: dict[str, dict] = {}
    for strat in strategies:
        cached = cache.get(qh, strat, model, PROMPT_SET_ID)
        if cached is not None:
            attempts[strat] = cached
            continue
        async with semaphore:
            t0 = time.time()
            try:
                attempt = await _run_strategy(
                    strat,
                    question=question, dataset=source_dataset,
                    gold_aliases=gold_aliases,
                    model=model, bm25_k=bm25_k,
                    ircot_max_iters=ircot_max_iters,
                    ircot_k_per_step=ircot_k_per_step,
                    client=client, retriever=retriever,
                )
                payload = attempt.to_json()
            except Exception as e:
                log.exception("Strategy %s failed on qid=%s: %s",
                              strat, row.get("question_id"), e)
                payload = Attempt(
                    strategy=strat,
                    pred_raw="", pred_extracted="",
                    em=0, f1=0.0, ok=False, latency_s=time.time() - t0,
                    prompt_tokens_est=0, completion_tokens_est=0,
                    error=f"{type(e).__name__}: {e}",
                ).to_json()
        cache.put(qh, strat, model, PROMPT_SET_ID, payload, created_at=time.time())
        attempts[strat] = payload

    # Pull cached attempts for canonical strategies not in --strategies, so
    # partial re-runs don't drop previously-cached results (and so cheapest-
    # wins labelling sees every strategy actually run, not just this slice).
    for strat in ALL_STRATEGIES:
        if strat in attempts:
            continue
        cached = cache.get(qh, strat, model, PROMPT_SET_ID)
        if cached is not None:
            attempts[strat] = cached

    oracle_label, label_source = label_mod.assign_label(
        attempts=attempts, source_dataset=source_dataset,
    )

    out = dict(row)
    out["dataset"] = source_dataset  # ensure present (some rows omit it)
    out["oracle_label"] = oracle_label
    out["label_source"] = label_source
    out["attempts"] = attempts
    out["labeller_model_id"] = model
    out["labeller_prompt_set"] = PROMPT_SET_ID
    out["labeller_commit"] = _git_commit()
    return out


# ---------------------------------------------------------------------------
# Per-(dataset, split) driver
# ---------------------------------------------------------------------------


async def annotate_split_dataset(
    *,
    split: str,
    dataset: str,
    rows: list[dict],
    strategies: tuple[str, ...],
    model: str,
    bm25_k: int,
    ircot_max_iters: int,
    ircot_k_per_step: int,
    cache: AttemptCache,
    concurrency: int,
    needs_retriever: bool,
) -> list[dict]:
    """Annotate one (split, dataset) shard. Returns the labelled rows."""

    client = get_async_client()
    retriever = None
    if needs_retriever:
        # Lazy — only load Pyserini when at least one strategy needs it.
        try:
            from src.retrieval import get_retriever
            retriever = get_retriever(dataset)
        except Exception as e:
            # Common failure modes: JAVA_HOME missing (JDK 21+ required),
            # index dir missing (run scripts/build_retrieval.py first).
            log.error(
                "Could not load BM25 retriever for %s: %s. "
                "Run scripts/build_retrieval.py to build the index (needs JDK 21).",
                dataset, e,
            )
            raise SystemExit(1)

    semaphore = asyncio.Semaphore(concurrency)
    tasks: list[asyncio.Task] = []
    for row in rows:
        tasks.append(asyncio.create_task(
            annotate_one_row(
                row, source_dataset=dataset, strategies=strategies,
                model=model, bm25_k=bm25_k,
                ircot_max_iters=ircot_max_iters,
                ircot_k_per_step=ircot_k_per_step,
                cache=cache, client=client, retriever=retriever,
                semaphore=semaphore,
            )
        ))

    done_rows: list[dict] = []
    t0 = time.time()
    for i, fut in enumerate(asyncio.as_completed(tasks), start=1):
        try:
            labeled = await fut
            done_rows.append(labeled)
        except Exception as e:
            log.exception("Row failed: %s", e)
            continue
        if i % 50 == 0 or i == len(rows):
            log.info(
                "  %s/%s  %4d/%d done  hit_rate=%.2f%%  elapsed=%.1fs",
                split, dataset, i, len(rows),
                100 * cache.hit_rate, time.time() - t0,
            )
    return done_rows


# ---------------------------------------------------------------------------
# Leakage filter
# ---------------------------------------------------------------------------


def filter_leaked(rows: list[dict], dataset: str) -> tuple[list[dict], int]:
    """Drop rows whose question_id appears in this dataset's prompt files."""
    try:
        leaks = prompts_mod.leaked_qids(dataset)
    except FileNotFoundError:
        return rows, 0
    if not leaks:
        return rows, 0
    keep, dropped = [], 0
    for r in rows:
        qid = str(r.get("question_id") or "")
        if qid in leaks:
            dropped += 1
        else:
            keep.append(r)
    return keep, dropped


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_labeled_jsonl(rows: list[dict], path: Path) -> None:
    """Append-then-dedup: write rows; final pass keeps the last entry per qid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    by_qid: dict[str, dict] = {}
    # If file already exists from a prior partial run, load it first so we keep
    # any qids we don't re-emit.
    if path.exists():
        for r in read_jsonl(path):
            qid = str(r.get("question_id") or "")
            if qid:
                by_qid[qid] = r
    for r in rows:
        qid = str(r.get("question_id") or "")
        if qid:
            by_qid[qid] = r
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf8") as f:
        for r in by_qid.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# QC report
# ---------------------------------------------------------------------------


def _hist(rows: list[dict], key: str) -> dict[str, int]:
    c: collections.Counter = collections.Counter()
    for r in rows:
        c[r.get(key, "")] += 1
    return dict(c)


def _strategy_pass_rates(rows: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for strat in ALL_STRATEGIES:
        ok = total = 0
        for r in rows:
            a = (r.get("attempts") or {}).get(strat)
            if a is None:
                continue
            total += 1
            if a.get("ok"):
                ok += 1
        out[strat] = (ok / total) if total else float("nan")
    return out


def _subset_check(rows: list[dict], a: str, b: str) -> dict[str, float]:
    """Of rows where strategy ``a`` passed, fraction where ``b`` also passed."""
    a_ok = a_and_b_ok = 0
    for r in rows:
        atts = r.get("attempts") or {}
        if not atts.get(a, {}).get("ok"):
            continue
        a_ok += 1
        if atts.get(b, {}).get("ok"):
            a_and_b_ok += 1
    return {
        f"n_{a}_ok": a_ok,
        f"frac_{a}_subset_{b}": (a_and_b_ok / a_ok) if a_ok else float("nan"),
    }


def build_qc_report(
    per_split: dict[str, dict[str, list[dict]]],
    *,
    cache: AttemptCache,
    n_dropped_leaks: dict[str, int],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "cache_hit_rate": cache.hit_rate,
        "cache_hits": cache.n_hits,
        "cache_misses": cache.n_misses,
        "prompt_set_id": PROMPT_SET_ID,
        "leakage_drops": n_dropped_leaks,
        "per_split": {},
    }
    for split, by_dataset in per_split.items():
        report["per_split"][split] = {}
        for ds, rows in by_dataset.items():
            report["per_split"][split][ds] = {
                "n": len(rows),
                "oracle_label_hist": _hist(rows, "oracle_label"),
                "label_source_hist": _hist(rows, "label_source"),
                "strategy_pass_rates": _strategy_pass_rates(rows),
                **_subset_check(rows, "no_retrieval", "single_step"),
                **_subset_check(rows, "single_step", "multi_step"),
            }
    return report


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _split_list(s: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not s:
        return default
    parts = tuple(p.strip() for p in s.split(",") if p.strip())
    return parts or default


async def run_async(args: argparse.Namespace) -> int:
    data_dir: Path = args.data_dir.resolve()
    labeled_dir = data_dir / "labeled"
    labeled_dir.mkdir(parents=True, exist_ok=True)

    # ---- resolve scope
    if args.dataset:
        datasets = (args.dataset,)
    else:
        datasets = _split_list(args.only_datasets, ALL_DATASETS)
    for d in datasets:
        if d not in ALL_DATASETS:
            log.error("Unknown dataset: %s. Choose from %s", d, ALL_DATASETS)
            return 2

    if args.split == "both":
        splits = SPLITS
    else:
        splits = (args.split,)

    strategies = _split_list(args.strategies, ALL_STRATEGIES)
    for s in strategies:
        if s not in ALL_STRATEGIES:
            log.error("Unknown strategy: %s. Choose from %s", s, ALL_STRATEGIES)
            return 2

    needs_retriever = any(s in ("single_step", "multi_step") for s in strategies)

    state = State(data_dir / ".annotate_state.json")
    if args.force:
        state.clear()

    log.info(
        "Plan: datasets=%s  splits=%s  strategies=%s  model=%s  "
        "concurrency=%d  bm25_k=%d  ircot_max_iters=%d  dry_run=%s",
        datasets, splits, strategies, args.model,
        args.concurrency, args.bm25_k, args.ircot_max_iters, args.dry_run,
    )

    cache = AttemptCache(args.cache_path or (data_dir / ".llm_cache.sqlite"))

    # Leakage filter — log drops per dataset (data_plan.md §2 risk register).
    n_dropped_leaks: dict[str, int] = {}
    for ds in datasets:
        try:
            leaks = prompts_mod.leaked_qids(ds)
        except FileNotFoundError as e:
            log.error("Vendored prompts missing for %s: %s", ds, e)
            return 1
        log.info("  %s: %d METADATA qids in prompt files", ds, len(leaks))

    per_split: dict[str, dict[str, list[dict]]] = {s: {} for s in splits}

    for split in splits:
        in_dir = data_dir / SPLIT_INPUT_DIR[split]
        for ds in datasets:
            in_path = in_dir / f"{ds}.jsonl"
            out_path = labeled_dir / split / f"{ds}.jsonl"
            state_key = f"{split}/{ds}/{','.join(sorted(strategies))}/{args.model}"

            if args.skip_existing and state.is_done(state_key):
                log.info("[%s/%s] skip — state.json says complete (%s)", split, ds, state_key)
                if out_path.exists():
                    per_split[split][ds] = read_jsonl(out_path)
                continue

            if not in_path.exists():
                log.error("Input missing: %s", in_path)
                return 1

            rows = read_jsonl(in_path)
            log.info("[%s/%s] read %d rows from %s", split, ds, len(rows), in_path)
            rows, dropped = filter_leaked(rows, ds)
            if dropped:
                log.info("[%s/%s] dropped %d leaked qids", split, ds, dropped)
            n_dropped_leaks[f"{split}/{ds}"] = dropped

            if args.max_questions is not None:
                rows = rows[: args.max_questions]
                log.info("[%s/%s] capped to %d rows (--max-questions)", split, ds, len(rows))

            if args.dry_run:
                log.info("[%s/%s] dry-run: would annotate %d rows × %d strategies",
                         split, ds, len(rows), len(strategies))
                continue

            t0 = time.time()
            labeled_rows = await annotate_split_dataset(
                split=split, dataset=ds, rows=rows, strategies=strategies,
                model=args.model, bm25_k=args.bm25_k,
                ircot_max_iters=args.ircot_max_iters,
                ircot_k_per_step=args.ircot_k_per_step,
                cache=cache, concurrency=args.concurrency,
                needs_retriever=needs_retriever,
            )
            elapsed = time.time() - t0
            log.info("[%s/%s] done %d rows in %.1fs (avg %.2fs/row)",
                     split, ds, len(labeled_rows), elapsed,
                     elapsed / max(1, len(labeled_rows)))

            # Per-class histogram so the user can spot collapse early.
            hist = collections.Counter(r["oracle_label"] for r in labeled_rows)
            log.info("[%s/%s] oracle_label histogram: %s", split, ds, dict(hist))

            write_labeled_jsonl(labeled_rows, out_path)
            per_split[split][ds] = labeled_rows
            state.mark_done(state_key)

    if args.dry_run:
        log.info("Dry-run complete. cache hit-rate so far: %.2f%%", 100 * cache.hit_rate)
        return 0

    # Build a complete per-split mapping from ALL on-disk shards (overlaying
    # this run's freshly-labelled rows). This makes the merge and QC report
    # span every dataset present on disk — not just the current --only-datasets
    # scope — so a subset run doesn't silently drop the other datasets' rows
    # from the merged classifier files. Only the splits we processed are
    # touched, so a `--split train` run leaves classifier_valid.jsonl alone.
    full_per_split = collect_full_per_split(splits, labeled_dir, per_split)

    # Merge per-split into the canonical classifier files.
    merged_paths: dict[str, Path] = {
        "train": labeled_dir / "classifier_train.jsonl",
        "eval":  labeled_dir / "classifier_valid.jsonl",
    }
    for split in splits:
        out = merged_paths[split]
        all_rows = collect_split_rows(split, labeled_dir, full_per_split)
        tmp = out.with_suffix(out.suffix + ".tmp")
        with open(tmp, "w", encoding="utf8") as f:
            for r in all_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, out)
        log.info("Wrote %d merged rows to %s", len(all_rows), out)

    # QC report
    report = build_qc_report(full_per_split, cache=cache, n_dropped_leaks=n_dropped_leaks)
    qc_path = labeled_dir / "qc_report.json"
    qc_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf8")
    log.info("Wrote QC report to %s", qc_path)
    log.info("Cache hit-rate: %.2f%%  (hits=%d, misses=%d)",
             100 * cache.hit_rate, cache.n_hits, cache.n_misses)

    return 0


def run(args: argparse.Namespace) -> int:
    try:
        return asyncio.run(run_async(args))
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        return 130


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dataset", choices=ALL_DATASETS,
                   help="Annotate just one dataset.")
    g.add_argument("--only-datasets", type=str,
                   help="Comma-separated subset of {hotpotqa,2wikimultihopqa,musique,nq,trivia,squad}. "
                        "Default: all six.")
    p.add_argument("--split", choices=("eval", "train", "both"), default="both",
                   help="Which input split to label. Default: both.")
    p.add_argument("--strategies", type=str, default=",".join(ALL_STRATEGIES),
                   help="Comma-separated subset of {no_retrieval,single_step,multi_step}.")
    p.add_argument("--model", type=str, default=LLM_MODEL,
                   help="Labeller LLM model id. Default: $LLM_MODEL from src/config.py.")
    p.add_argument("--bm25-k", type=int, default=15,
                   help="BM25 top-k for single_step. Default 15.")
    p.add_argument("--ircot-max-iters", type=int, default=4,
                   help="Max IRCoT iterations for multi_step. Default 4.")
    p.add_argument("--ircot-k-per-step", type=int, default=6,
                   help="BM25 top-k per IRCoT step. Default 6.")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Async semaphore size — how many in-flight LLM calls. Default 4.")
    p.add_argument("--max-questions", type=int, default=None,
                   help="Cap rows per (split,dataset) — for smoke tests.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                   help="Project data root. Default: <repo>/data")
    p.add_argument("--cache-path", type=Path, default=None,
                   help="Override the SQLite cache path. Default: <data-dir>/.llm_cache.sqlite")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="(default) Skip (split,dataset,strategies,model) combos already done.")
    p.add_argument("--force", action="store_true",
                   help="Clear .annotate_state.json — re-runs every shard.")
    p.add_argument("--dry-run", action="store_true",
                   help="Read inputs, log what would happen, write nothing.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
