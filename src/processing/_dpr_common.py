"""Shared NQ/Trivia/SQuAD DPR processing.

The upstream ``process_{nq,trivia,squad}.py`` are byte-for-byte identical
except for the dataset name string and the input filename pattern. We
consolidate the body here and the per-dataset modules wire the rest.

Source:
- https://github.com/starsuzi/Adaptive-RAG/blob/main/processing_scripts/process_nq.py
- https://github.com/starsuzi/Adaptive-RAG/blob/main/processing_scripts/process_trivia.py
- https://github.com/starsuzi/Adaptive-RAG/blob/main/processing_scripts/process_squad.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm


def safe_sample(items: list[Any], count: int) -> list[Any]:
    count = min(count, len(items))
    return random.sample(items, count) if count > 0 else []


def write_dpr_instances_to_filepath(
    raw_instances: Iterable[dict],
    output_filepath: Path,
    set_name: str,
    dataset_label: str,
    qid_prefix: str,
) -> int:
    """Mirror of upstream ``write_{nq,trivia,squad}_instances_to_filepath``.

    ``dataset_label`` is what appears in ``processed_instance["dataset"]``
    (e.g. ``"nq"``). ``qid_prefix`` is the question_id prefix (e.g.
    ``"single_nq_"``). The original code spelled these out per-file.
    """

    print(f"Writing in: {output_filepath}")
    raw_list = list(raw_instances)
    print(len(raw_list))

    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with open(output_filepath, "w", encoding="utf8") as output_file:

        for idx, raw_instance in tqdm(enumerate(raw_list)):

            # Generic RC Format
            processed_instance: dict = {}
            processed_instance["dataset"] = dataset_label
            processed_instance["question_id"] = qid_prefix + set_name + "_" + str(idx)
            processed_instance["question_text"] = raw_instance["question"]

            answers_object = {
                "number": "",
                "date": {"day": "", "month": "", "year": ""},
                "spans": raw_instance["answers"],
            }

            processed_instance["answers_objects"] = [answers_object]

            lst_context: list[dict] = []
            context_id = 0

            for ctx in raw_instance["positive_ctxs"]:
                lst_context.append({
                    "idx": context_id,
                    "title": ctx["title"].strip(),
                    "paragraph_text": ctx["text"].strip(),
                    "is_supporting": True,
                })
                context_id += 1

            sampled_neg = safe_sample(raw_instance["negative_ctxs"], 5)
            for ctx in sampled_neg:
                lst_context.append({
                    "idx": context_id,
                    "title": ctx["title"].strip(),
                    "paragraph_text": ctx["text"].strip(),
                    "is_supporting": False,
                })
                context_id += 1

            sampled_hard = safe_sample(raw_instance["hard_negative_ctxs"], 5)
            for ctx in sampled_hard:
                lst_context.append({
                    "idx": context_id,
                    "title": ctx["title"].strip(),
                    "paragraph_text": ctx["text"].strip(),
                    "is_supporting": False,
                })
                context_id += 1

            processed_instance["contexts"] = lst_context

            output_file.write(json.dumps(processed_instance) + "\n")
            n_written += 1

    return n_written
