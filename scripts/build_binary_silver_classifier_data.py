from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.annotate_lib.label import binary_fallback_for


DATASETS = ("musique", "hotpotqa", "2wikimultihopqa", "nq", "trivia", "squad")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def processed_row_to_binary_label(row: dict[str, Any], dataset: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "question_id": str(row["question_id"]),
        "question_text": row["question_text"],
        "oracle_label": binary_fallback_for(dataset),
        "label_source": "binary_inductive_bias",
    }


def silver_row_for_training(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": row["dataset"],
        "question_id": str(row["question_id"]),
        "question_text": row["question_text"],
        "oracle_label": row["oracle_label"],
        "label_source": row.get("label_source", "silver"),
    }


def adaptive_binary_row_for_training(row: dict[str, Any]) -> dict[str, Any]:
    label = {"A": "no_retrieval", "B": "single_step", "C": "multi_step"}[row["answer"]]
    return {
        "dataset": row["dataset_name"],
        "question_id": str(row["id"]),
        "question_text": row["question"],
        "oracle_label": label,
        "label_source": "binary_inductive_bias",
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build an Adaptive-RAG-style binary+silver classifier training file. "
            "Binary examples come from processed train rows with dataset-level "
            "inductive-bias labels; silver examples come from annotated labels."
        )
    )
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument(
        "--adaptive-binary-json",
        type=Path,
        default=None,
        help=(
            "Optional Adaptive-RAG classifier/data/.../binary/total_data_train.json. "
            "When set, this is used instead of --processed-dir."
        ),
    )
    p.add_argument("--silver-train-file", type=Path, default=Path("data/labeled/classifier_train.jsonl"))
    p.add_argument("--output-file", type=Path, default=Path("data/labeled/classifier_train_binary_silver.jsonl"))
    p.add_argument("--binary-per-dataset", type=int, default=400)
    p.add_argument("--only-datasets", type=str, default=",".join(DATASETS))
    p.add_argument(
        "--include-fallback-silver",
        action="store_true",
        help=(
            "Include rows whose label_source is binary_fallback. By default, "
            "only label_source=silver rows are appended, matching Adaptive-RAG's "
            "silver file more closely."
        ),
    )
    p.add_argument(
        "--allow-missing-datasets",
        action="store_true",
        help="Skip datasets whose processed train file is missing instead of failing.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    datasets = tuple(d.strip() for d in args.only_datasets.split(",") if d.strip())

    silver_rows_raw = read_jsonl(args.silver_train_file)
    if not args.include_fallback_silver:
        silver_rows_raw = [
            row for row in silver_rows_raw
            if row.get("label_source") == "silver"
        ]
    silver_ids = {str(row["question_id"]) for row in silver_rows_raw}
    silver_rows = [silver_row_for_training(row) for row in silver_rows_raw]

    binary_rows: list[dict[str, Any]] = []
    per_dataset_counts: dict[str, int] = {}
    missing: list[Path] = []

    if args.adaptive_binary_json is not None:
        with open(args.adaptive_binary_json, "r", encoding="utf8") as fh:
            adaptive_rows = json.load(fh)
        for row in adaptive_rows:
            converted = adaptive_binary_row_for_training(row)
            if converted["question_id"] in silver_ids:
                continue
            binary_rows.append(converted)
            per_dataset_counts[converted["dataset"]] = per_dataset_counts.get(converted["dataset"], 0) + 1
    else:
        for dataset in datasets:
            train_path = args.processed_dir / dataset / "train.jsonl"
            if not train_path.exists():
                missing.append(train_path)
                if args.allow_missing_datasets:
                    continue
                print(f"Missing processed train file: {train_path}")
                return 1

            kept = 0
            for row in read_jsonl(train_path):
                qid = str(row["question_id"])
                if qid in silver_ids:
                    continue
                binary_rows.append(processed_row_to_binary_label(row, dataset))
                kept += 1
                if kept >= args.binary_per_dataset:
                    break
            per_dataset_counts[dataset] = kept

    out_rows = binary_rows + silver_rows
    write_jsonl(args.output_file, out_rows)

    print(f"silver rows: {len(silver_rows)}")
    print(f"binary rows: {len(binary_rows)}")
    print(f"total rows: {len(out_rows)}")
    print(f"per-dataset binary counts: {per_dataset_counts}")
    if missing:
        print("missing processed train files:")
        for path in missing:
            print(f"  {path}")
    print(f"wrote: {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
