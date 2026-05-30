"""Process HotpotQA raw JSON into unified per-instance JSONL.

Port of: https://github.com/starsuzi/Adaptive-RAG/blob/main/processing_scripts/process_hotpotqa.py

DIVERGENCE FROM UPSTREAM: the original script uses
``datasets.load_dataset("hotpot_qa", "distractor")``. Our port reads the raw
JSON files directly (``hotpot_train_v1.1.json`` / ``hotpot_dev_distractor_v1.json``)
that ``data_plan.md`` and ``download/raw_data.sh`` point at. The resulting
``processed_instance`` dict has the same keys and semantics — only the input
format (raw upstream JSON list-of-pairs) differs from HF's nested-dict shape.
Output rows match the upstream byte-for-byte up to dict key ordering.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from src.processing._lib import read_json


def _write_hotpotqa_instances_to_filepath(instances: Iterable[dict], full_filepath: Path) -> int:
    """Mirror of upstream ``write_hotpotqa_instances_to_filepath`` adapted to the
    raw-JSON ``context`` shape (list of ``[title, [sentence, ...]]`` pairs).
    """

    max_num_tokens = 1000  # clip later.

    hop_sizes: Counter = Counter()
    n_written = 0
    print(f"Writing in: {full_filepath}")
    full_filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(full_filepath, "w", encoding="utf8") as full_file:
        for raw_instance in tqdm(instances):

            # Generic RC Format
            processed_instance: dict = {}
            processed_instance["dataset"] = "hotpotqa"
            processed_instance["question_id"] = raw_instance["_id"]
            processed_instance["question_text"] = raw_instance["question"]
            processed_instance["level"] = raw_instance["level"]
            processed_instance["type"] = raw_instance["type"]

            answers_object = {
                "number": "",
                "date": {"day": "", "month": "", "year": ""},
                "spans": [raw_instance["answer"]],
            }
            processed_instance["answers_objects"] = [answers_object]

            # raw_instance["context"] is List[ [title, [sentence_str, ...]] ]
            # raw_instance["supporting_facts"] is List[ [title, sent_id] ]
            raw_context = raw_instance["context"]
            supporting_titles = [sf[0] for sf in raw_instance["supporting_facts"]]

            title_to_paragraph: dict[str, str] = {
                title: "".join(sentences) for title, sentences in raw_context
            }
            paragraph_to_title: dict[str, str] = {
                "".join(sentences): title for title, sentences in raw_context
            }

            gold_paragraph_texts = [title_to_paragraph[title] for title in supporting_titles
                                    if title in title_to_paragraph]
            gold_paragraph_texts = set(gold_paragraph_texts)

            paragraph_texts = ["".join(sentences) for _, sentences in raw_context]
            paragraph_texts = list(set(paragraph_texts))

            processed_instance["contexts"] = [
                {
                    "idx": index,
                    "title": paragraph_to_title[paragraph_text].strip(),
                    "paragraph_text": paragraph_text.strip(),
                    "is_supporting": paragraph_text in gold_paragraph_texts,
                }
                for index, paragraph_text in enumerate(paragraph_texts)
            ]

            supporting_contexts = [c for c in processed_instance["contexts"] if c["is_supporting"]]
            hop_sizes[len(supporting_contexts)] += 1

            for context in processed_instance["contexts"]:
                context["paragraph_text"] = " ".join(context["paragraph_text"].split(" ")[:max_num_tokens])

            full_file.write(json.dumps(processed_instance) + "\n")
            n_written += 1

    print(f"Hop-sizes: {str(hop_sizes)}")
    return n_written


def main(input_dir: Path, output_dir: Path) -> dict[str, int]:
    """Process HotpotQA train + dev into ``{output_dir}/{train,dev}.jsonl``.

    Expects ``input_dir`` to contain ``hotpot_train_v1.1.json`` and
    ``hotpot_dev_distractor_v1.json``.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}

    train_in = input_dir / "hotpot_train_v1.1.json"
    train_out = output_dir / "train.jsonl"
    print(f"Processing train: {train_in}")
    counts["train"] = _write_hotpotqa_instances_to_filepath(read_json(train_in), train_out)

    dev_in = input_dir / "hotpot_dev_distractor_v1.json"
    dev_out = output_dir / "dev.jsonl"
    print(f"Processing dev: {dev_in}")
    counts["dev"] = _write_hotpotqa_instances_to_filepath(read_json(dev_in), dev_out)

    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Directory with hotpot_train_v1.1.json and hotpot_dev_distractor_v1.json")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output directory; will write {train,dev}.jsonl")
    args = parser.parse_args()
    main(args.input, args.output)
