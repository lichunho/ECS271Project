"""Regression tests for the merge/QC assembly in scripts/annotate.py.

The merge block used to build each canonical classifier file from
``per_split[split]``, which is only populated for datasets in the current
run's ``--only-datasets`` scope. A subset run therefore rewrote the merged
file with only that subset, silently dropping the other datasets' shards.

``collect_split_rows`` fixes this by taking the union of all on-disk shards
for a split (preferring this run's freshly-labelled rows when present). These
tests pin that behaviour.

Run with::

    .\\.venv\\Scripts\\python.exe -m pytest tests\\test_annotate_merge.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the repo root is on sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import annotate as ann


def _write_shard(labeled_dir: Path, split: str, ds: str, rows: list[dict]) -> None:
    path = labeled_dir / split / f"{ds}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_collect_split_rows_unions_fresh_and_on_disk(tmp_path):
    """A subset run that only re-labelled ``nq`` must still pick up the
    on-disk ``squad`` shard, and the freshly-labelled ``nq`` rows must win
    over whatever happens to be on disk for ``nq``.

    Uses real dataset names (``nq``, ``squad``) so they are included by the
    helper's default ``datasets=ALL_DATASETS``.
    """
    labeled_dir = tmp_path / "labeled"

    # On-disk shards for both datasets. The on-disk nq rows are *stale* — the
    # fresh per_split entry below should supersede them.
    _write_shard(labeled_dir, "train", "nq", [
        {"question_id": "nq-stale-1", "oracle_label": "single_step"},
    ])
    _write_shard(labeled_dir, "train", "squad", [
        {"question_id": "squad-1", "oracle_label": "no_retrieval"},
        {"question_id": "squad-2", "oracle_label": "single_step"},
    ])

    fresh_nq = [
        {"question_id": "nq-fresh-1", "oracle_label": "no_retrieval"},
        {"question_id": "nq-fresh-2", "oracle_label": "single_step"},
    ]

    rows = ann.collect_split_rows(
        "train", labeled_dir, per_split={"train": {"nq": fresh_nq}},
    )

    qids = {r["question_id"] for r in rows}
    # squad came from disk ...
    assert "squad-1" in qids
    assert "squad-2" in qids
    # ... and nq came from the fresh per_split entry (stale on-disk nq dropped).
    assert "nq-fresh-1" in qids
    assert "nq-fresh-2" in qids
    assert "nq-stale-1" not in qids
    assert len(rows) == 4


def test_collect_split_rows_reads_disk_only_when_no_per_split(tmp_path):
    """With no ``per_split`` at all, every dataset is sourced from disk."""
    labeled_dir = tmp_path / "labeled"
    _write_shard(labeled_dir, "train", "nq", [{"question_id": "nq-1"}])
    _write_shard(labeled_dir, "train", "squad", [{"question_id": "squad-1"}])

    rows = ann.collect_split_rows("train", labeled_dir)
    qids = {r["question_id"] for r in rows}
    assert qids == {"nq-1", "squad-1"}


def test_collect_split_rows_skips_missing_datasets(tmp_path):
    """Datasets with neither fresh rows nor an on-disk shard are skipped."""
    labeled_dir = tmp_path / "labeled"
    _write_shard(labeled_dir, "train", "nq", [{"question_id": "nq-1"}])

    rows = ann.collect_split_rows("train", labeled_dir)
    assert [r["question_id"] for r in rows] == ["nq-1"]


def test_collect_full_per_split_spans_all_on_disk(tmp_path):
    """``collect_full_per_split`` overlays fresh rows on the on-disk shards
    and only covers the splits requested (leaving other splits untouched)."""
    labeled_dir = tmp_path / "labeled"
    _write_shard(labeled_dir, "train", "nq", [{"question_id": "nq-disk"}])
    _write_shard(labeled_dir, "train", "squad", [{"question_id": "squad-disk"}])
    # An eval shard exists on disk but eval is NOT in the requested splits.
    _write_shard(labeled_dir, "eval", "nq", [{"question_id": "nq-eval"}])

    fresh = {"train": {"nq": [{"question_id": "nq-fresh"}]}}
    full = ann.collect_full_per_split(["train"], labeled_dir, fresh)

    assert set(full.keys()) == {"train"}  # eval untouched
    assert set(full["train"].keys()) == {"nq", "squad"}
    assert [r["question_id"] for r in full["train"]["nq"]] == ["nq-fresh"]
    assert [r["question_id"] for r in full["train"]["squad"]] == ["squad-disk"]
