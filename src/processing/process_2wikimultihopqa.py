"""Process 2WikiMultiHopQA raw JSON into unified per-instance JSONL.

Port of: https://github.com/starsuzi/Adaptive-RAG/blob/main/processing_scripts/process_2wikimultihopqa.py

Logic preserved verbatim from upstream. Input/output paths were hard-coded;
this version accepts them as arguments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.processing._lib import read_json, write_jsonl


def main(input_dir: Path, output_dir: Path) -> dict[str, int]:
    """Process 2Wiki train + dev into ``{output_dir}/{train,dev}.jsonl``.

    Expects ``input_dir`` to contain ``{train,dev}.json``.
    """
    set_names = ["train", "dev"]

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}

    for set_name in set_names:
        print(f"Processing {set_name}")

        processed_instances = []

        input_filepath = input_dir / f"{set_name}.json"
        output_filepath = output_dir / f"{set_name}.jsonl"

        raw_instances = read_json(input_filepath)

        for raw_instance in raw_instances:

            question_id = raw_instance["_id"]
            question_text = raw_instance["question"]
            raw_contexts = raw_instance["context"]

            supporting_titles = list(set([e[0] for e in raw_instance["supporting_facts"]]))

            evidences = raw_instance["evidences"]
            reasoning_steps = [" ".join(evidence) for evidence in evidences]

            processed_contexts = []
            for index, raw_context in enumerate(raw_contexts):
                title = raw_context[0]
                paragraph_text = " ".join(raw_context[1]).strip()
                is_supporting = title in supporting_titles
                processed_contexts.append(
                    {
                        "idx": index,
                        "title": title.strip(),
                        "paragraph_text": paragraph_text,
                        "is_supporting": is_supporting,
                    }
                )

            answers_object = {
                "number": "",
                "date": {"day": "", "month": "", "year": ""},
                "spans": [raw_instance["answer"]],
            }
            answers_objects = [answers_object]

            processed_instance = {
                "question_id": question_id,
                "question_text": question_text,
                "answers_objects": answers_objects,
                "contexts": processed_contexts,
                "reasoning_steps": reasoning_steps,
            }

            processed_instances.append(processed_instance)

        write_jsonl(processed_instances, output_filepath)
        counts[set_name] = len(processed_instances)

    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    main(args.input, args.output)
