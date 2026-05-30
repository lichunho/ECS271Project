"""Process TriviaQA DPR-curated JSON into unified per-instance JSONL.

Port of: https://github.com/starsuzi/Adaptive-RAG/blob/main/processing_scripts/process_trivia.py

Logic preserved verbatim from upstream, including ``random.seed(13370)`` set
before any sampling.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.processing._dpr_common import write_dpr_instances_to_filepath
from src.processing._lib import read_json


def main(input_dir: Path, output_dir: Path) -> dict[str, int]:
    """Process Trivia train + dev into ``{output_dir}/{train,dev}.jsonl``.

    Expects ``input_dir`` to contain ``biencoder-trivia-{train,dev}.json``.
    """
    random.seed(13370)  # Don't change. Match upstream module-level seed.

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}

    train_in = input_dir / "biencoder-trivia-train.json"
    train_out = output_dir / "train.jsonl"
    counts["train"] = write_dpr_instances_to_filepath(
        read_json(train_in), train_out, "train",
        dataset_label="trivia", qid_prefix="single_trivia_",
    )

    dev_in = input_dir / "biencoder-trivia-dev.json"
    dev_out = output_dir / "dev.jsonl"
    counts["dev"] = write_dpr_instances_to_filepath(
        read_json(dev_in), dev_out, "dev",
        dataset_label="trivia", qid_prefix="single_trivia_",
    )

    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    main(args.input, args.output)
