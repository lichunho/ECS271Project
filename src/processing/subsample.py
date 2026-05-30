"""Subsample 500 instances per (dataset, split).

Port of: https://github.com/starsuzi/Adaptive-RAG/blob/main/processing_scripts/subsample_dataset_and_remap_paras.py

Reproducibility notes from the upstream code:
- ``random.seed(13370)`` is set at module import time *before* any sampling.
  We do the same, then re-seed inside ``main`` so the per-call sample is
  deterministic regardless of import order.
- ``sample_size = 500`` is hard-coded for ``set_name == "test"``. For
  ``dev_diff_size`` the size comes from the CLI argument.
- For ``test``, the upstream code optionally avoids ``dev_subsampled.jsonl``
  ONLY if it already exists. In a clean run that file does not exist, so
  ``test`` samples from the full ``dev.jsonl`` — that's what produced the
  reference ``test_subsampled.jsonl`` shipped in the repo tarball. We mirror
  that behavior.
- For ``dev_diff_size``, the upstream code avoids ``test_subsampled.jsonl``.
  This means ``test`` MUST be subsampled BEFORE ``dev_diff_size``.

PARAGRAPH REMAP (opt-in): upstream calls ``find_matching_paragraph_text``
against its Elasticsearch retriever to rewrite ``context[*].{title,
paragraph_text}`` so the saved fields match the canonical corpus text the
inference-time retriever will return. We replicate that behavior via an
injected callable (``find_paragraph_callable``). Pass ``None`` (the default)
to skip — the qid set is identical either way, so the step-10/11 diff
passes regardless. After ``scripts/build_retrieval.py`` has built the BM25
indices you can re-run this with ``src.retrieval.find_matching_paragraph_text``
to align paragraph text with the corpus.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from src.processing._lib import read_jsonl, write_jsonl

log = logging.getLogger(__name__)

random.seed(13370)  # Don't change this. Matches upstream module-level seed.


# Type alias: the signature must match
# ``src.retrieval.find_matching_paragraph_text(corpus_name, paragraph_text)``,
# returning ``{"title": ..., "paragraph_text": ...}`` or ``None``.
FindParagraphCallable = Callable[[str, str], dict[str, str] | None]


def main(
    dataset_name: str,
    set_name: str,
    input_dir: Path,
    output_dir: Path,
    sample_size: int = 500,
    *,
    find_paragraph_callable: FindParagraphCallable | None = None,
) -> Path:
    """Subsample 500 instances from ``input_dir/dev.jsonl``.

    Args:
        dataset_name: one of hotpotqa/2wikimultihopqa/musique/nq/trivia/squad.
        set_name: ``test`` or ``dev_diff_size``.
        input_dir: directory containing ``dev.jsonl`` and (for ``dev_diff_size``)
                   ``test_subsampled.jsonl``.
        output_dir: where to write the subsampled file (usually same as input_dir).
        sample_size: number of instances. Default 500; only used by ``dev_diff_size``.
                     For ``test`` the upstream code hard-codes 500.

    Returns the output file path.
    """
    if set_name not in ("test", "dev_diff_size"):
        raise ValueError(f"set_name must be 'test' or 'dev_diff_size', got {set_name!r}")

    # Re-seed so order of dataset processing doesn't change the sample.
    random.seed(13370)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    avoid_question_ids_file_path: Path | None = None

    if set_name == "test":
        # Upstream: avoid dev_subsampled.jsonl only if it already exists. In
        # a clean run it doesn't, matching the reference tarball.
        candidate = input_dir / "dev_subsampled.jsonl"
        avoid_question_ids_file_path = candidate if candidate.exists() else None
        effective_sample_size = 500  # upstream hard-codes this

    else:  # dev_diff_size
        avoid_question_ids_file_path = input_dir / "test_subsampled.jsonl"
        effective_sample_size = sample_size

    input_file_path = input_dir / "dev.jsonl"
    instances = read_jsonl(input_file_path)

    if avoid_question_ids_file_path is not None:
        if not avoid_question_ids_file_path.exists():
            raise FileNotFoundError(
                f"Need {avoid_question_ids_file_path} to exist before subsampling "
                f"{set_name} for {dataset_name}. Run 'test' subsample first."
            )
        avoid_ids = set(
            inst["question_id"] for inst in read_jsonl(avoid_question_ids_file_path)
        )
        instances = [inst for inst in instances if inst["question_id"] not in avoid_ids]

    instances = random.sample(instances, effective_sample_size)

    # Paragraph remap pass — port of upstream's loop. Only runs when the
    # caller supplies ``find_paragraph_callable``; this keeps the
    # data-sourcing step decoupled from a live retriever (the sourcing
    # orchestrator's default is None, so step-10 diffs still pass).
    if find_paragraph_callable is not None and set_name == "test":
        n_remapped = 0
        n_missed = 0
        n_skipped = 0
        for instance in tqdm(instances, desc=f"{dataset_name}/{set_name} remap"):
            for context in instance["contexts"]:
                if context in instance.get("pinned_contexts", []):
                    # Pinned contexts (iirc main) aren't in the associated
                    # wikipedia corpus — skip, same as upstream.
                    n_skipped += 1
                    continue
                if dataset_name in ("nq", "trivia", "squad"):
                    # Upstream comments this branch out
                    # (``find_matching_paragraph_text('wiki', ...)``);
                    # we mirror that — the wiki BM25 corpus is 100-word
                    # passages, so a fuzz>95 match against the long
                    # DPR-positive paragraphs almost never succeeds.
                    n_skipped += 1
                    continue
                retrieved = find_paragraph_callable(
                    dataset_name, context["paragraph_text"]
                )
                if retrieved is None:
                    n_missed += 1
                    continue
                context["title"] = retrieved["title"]
                context["paragraph_text"] = retrieved["paragraph_text"]
                n_remapped += 1
        log.info("%s/%s remap: %d remapped, %d below-threshold, %d skipped",
                 dataset_name, set_name, n_remapped, n_missed, n_skipped)

    if set_name == "dev_diff_size":
        output_file_path = output_dir / f"dev_{sample_size}_subsampled.jsonl"
    else:
        output_file_path = output_dir / f"{set_name}_subsampled.jsonl"

    write_jsonl(instances, output_file_path)
    return output_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_name", choices=(
        "hotpotqa", "2wikimultihopqa", "musique", "nq", "trivia", "squad",
    ))
    parser.add_argument("set_name", choices=("test", "dev_diff_size"))
    parser.add_argument("sample_size", type=int, nargs="?", default=500)
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Directory containing dev.jsonl")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write the subsampled file (default = input-dir)")
    parser.add_argument("--remap-paragraphs", action="store_true",
                        help="Run the BM25 paragraph remap pass (matches "
                             "upstream behaviour). Requires the dataset's "
                             "Lucene index to exist — run scripts/build_retrieval.py "
                             "first. No-op for nq/trivia/squad and for "
                             "set_name=dev_diff_size.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    callable_: FindParagraphCallable | None = None
    if args.remap_paragraphs:
        from src.retrieval import find_matching_paragraph_text
        callable_ = find_matching_paragraph_text

    output_dir = args.output_dir or args.input_dir
    out = main(
        args.dataset_name, args.set_name, args.input_dir, output_dir,
        args.sample_size, find_paragraph_callable=callable_,
    )
    print(f"Wrote: {out}")
