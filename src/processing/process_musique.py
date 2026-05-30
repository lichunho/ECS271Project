"""Process MuSiQue raw JSONL into unified per-instance JSONL.

Port of: https://github.com/starsuzi/Adaptive-RAG/blob/main/processing_scripts/process_musique.py

Logic preserved verbatim from upstream. We use ``musique_ans_v1.0_*.jsonl``
(the "Ans" variant) as the data plan specifies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.processing._lib import read_jsonl, write_jsonl


def main(input_dir: Path, output_dir: Path) -> dict[str, int]:
    """Process MuSiQue train + dev into ``{output_dir}/{train,dev}.jsonl``.

    Expects ``input_dir`` to contain ``musique_ans_v1.0_{train,dev}.jsonl``.
    """
    set_names = ["train", "dev"]

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}

    for set_name in set_names:
        processed_instances = []

        input_filepath = input_dir / f"musique_ans_v1.0_{set_name}.jsonl"
        output_filepath = output_dir / f"{set_name}.jsonl"

        raw_instances = read_jsonl(input_filepath)

        for raw_instance in raw_instances:

            answers_object = {
                "number": "",
                "date": {"day": "", "month": "", "year": ""},
                "spans": [raw_instance["answer"]],
            }

            number_to_answer: dict[int, str] = {}
            sentences: list[str] = []
            for index, reasoning_step in enumerate(raw_instance["question_decomposition"]):
                number = index + 1
                question = reasoning_step["question"]
                for mentioned_number in range(1, 10):
                    if f"#{mentioned_number}" in reasoning_step["question"]:
                        if mentioned_number not in number_to_answer:
                            print("WARNING: mentioned_number not present in number_to_answer.")
                        else:
                            question = question.replace(f"#{mentioned_number}", number_to_answer[mentioned_number])
                answer = reasoning_step["answer"]
                number_to_answer[number] = answer
                sentence = " >>>> ".join([question.strip(), answer.strip()])
                sentences.append(sentence)

            processed_instance = {
                "question_id": raw_instance["id"],
                "question_text": raw_instance["question"],
                "contexts": [
                    {
                        "idx": index,
                        "paragraph_text": paragraph["paragraph_text"].strip(),
                        "title": paragraph["title"].strip(),
                        "is_supporting": paragraph["is_supporting"],
                    }
                    for index, paragraph in enumerate(raw_instance["paragraphs"])
                ],
                "answers_objects": [answers_object],
                "reasoning_steps": sentences,
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
