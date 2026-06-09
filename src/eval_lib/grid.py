"""Load the labelled grid: ``data/labeled/{split}/{dataset}.jsonl`` -> {qid: GridRow}.

Each row carries the three strategies' precomputed attempts plus the oracle
label and its source. The grid is the single source of truth for every metric;
executing route R on question q yields exactly ``attempts[R]`` (same model
labels and answers — see CLAUDE.md "Key invariants").
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# The six datasets that make up a split, in a stable display order.
DATASETS = (
    "nq",
    "trivia",
    "squad",
    "hotpotqa",
    "2wikimultihopqa",
    "musique",
)

ROUTES = ("no_retrieval", "single_step", "multi_step")


@dataclass(frozen=True)
class Attempt:
    """One strategy's precomputed result for a question (mirrors the on-disk
    ``attempts[strategy]`` object; only the fields the harness aggregates)."""

    strategy: str
    em: int
    f1: float
    ok: bool
    latency_s: float
    prompt_tokens_est: int
    completion_tokens_est: int
    n_hops: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens_est + self.completion_tokens_est

    @classmethod
    def from_dict(cls, d: dict) -> "Attempt":
        return cls(
            strategy=d["strategy"],
            em=int(d.get("em", 0)),
            f1=float(d.get("f1", 0.0)),
            ok=bool(d.get("ok", False)),
            latency_s=float(d.get("latency_s", 0.0)),
            prompt_tokens_est=int(d.get("prompt_tokens_est", 0)),
            completion_tokens_est=int(d.get("completion_tokens_est", 0)),
            n_hops=int(d.get("n_hops", 0)),
        )


@dataclass(frozen=True)
class GridRow:
    question_id: str
    dataset: str
    oracle_label: str
    label_source: str  # "silver" | "binary_fallback"
    attempts: dict  # {route: Attempt}

    @property
    def is_silver(self) -> bool:
        return self.label_source == "silver"


def _row_to_grid(row: dict) -> GridRow:
    attempts = {
        route: Attempt.from_dict(att)
        for route, att in row["attempts"].items()
    }
    return GridRow(
        question_id=str(row["question_id"]),
        dataset=row["dataset"],
        oracle_label=row["oracle_label"],
        label_source=row["label_source"],
        attempts=attempts,
    )


def load_grid(labeled_dir: Path, split: str, datasets=None) -> dict:
    """Load ``{labeled_dir}/{split}/{dataset}.jsonl`` for each dataset into a
    ``{question_id: GridRow}`` map. Raises on a duplicate qid (the data
    guarantees uniqueness) or a missing shard file."""
    datasets = tuple(datasets) if datasets else DATASETS
    split_dir = Path(labeled_dir) / split
    grid: dict = {}
    for ds in datasets:
        path = split_dir / f"{ds}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing labelled shard: {path}")
        with open(path, "r", encoding="utf8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                gr = _row_to_grid(json.loads(line))
                if gr.question_id in grid:
                    raise ValueError(
                        f"duplicate question_id {gr.question_id!r} "
                        f"(first in {grid[gr.question_id].dataset}, again in {ds})"
                    )
                grid[gr.question_id] = gr
    return grid
