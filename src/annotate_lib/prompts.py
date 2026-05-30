"""Load + compose the vendored Adaptive-RAG few-shot prompts.

The files under ``prompts/{dataset}/*.txt`` are verbatim copies of
``starsuzi/Adaptive-RAG/prompts/{dataset}/*_flan_t5.txt`` (keeping the
upstream filenames so future readers can ``diff`` them). They are
completion-style few-shot prompts: a series of ``# METADATA: ...`` +
``Q: ...`` + ``A: ...`` demonstrations, blank-line separated, with no
trailing answer for the test question.

We follow Adaptive-RAG's ``StepByStepCOTGenParticipant.query`` exactly::

    test_example_str = context + "\\n\\n" + f"Q: {question}" + "\\n" + f"A: {generation_so_far}"
    prompt = "\\n\\n\\n".join([self.prompt, test_example_str]).strip()

where ``question`` is the dataset-typed instruction line plus the user's
question, and ``context`` is ``\\n\\n``-joined ``"Wikipedia Title:
{title}\\n{paragraph_text}"`` blocks. For no-context strategies,
``context`` is the empty string.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Sequence


# Per data_plan.md: CoT for multi-hop; direct for the rest.
COT_DATASETS: frozenset[str] = frozenset({"hotpotqa", "2wikimultihopqa", "musique"})
DIRECT_DATASETS: frozenset[str] = frozenset({"nq", "trivia", "squad"})
ALL_DATASETS: tuple[str, ...] = (
    "hotpotqa", "2wikimultihopqa", "musique", "nq", "trivia", "squad",
)

# Version tag baked into the cache key — bump when prompts files change.
PROMPT_SET_ID: str = "flan_t5_v1"

_PROMPT_ROOT = Path(__file__).resolve().parent.parent.parent / "prompts"


def _filename(dataset: str, *, with_context: bool) -> str:
    """Vendored filename for the appropriate flan_t5 variant."""
    style = "cot" if dataset in COT_DATASETS else "direct"
    if with_context:
        return f"gold_with_1_distractors_context_{style}_qa_flan_t5.txt"
    return f"no_context_{style}_qa_flan_t5.txt"


def prompt_path(dataset: str, *, with_context: bool) -> Path:
    if dataset not in ALL_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    return _PROMPT_ROOT / dataset / _filename(dataset, with_context=with_context)


@lru_cache(maxsize=32)
def load_prompt(dataset: str, *, with_context: bool) -> str:
    """Read and cache one few-shot prompt file (verbatim)."""
    path = prompt_path(dataset, with_context=with_context)
    if not path.exists():
        raise FileNotFoundError(
            f"Vendored prompt missing: {path}. Re-run the prompt-vendoring step in "
            "scripts/annotate.py's README."
        )
    return path.read_text(encoding="utf8").rstrip()


def is_cot(dataset: str) -> bool:
    return dataset in COT_DATASETS


def _instruction(dataset: str) -> str:
    if dataset in COT_DATASETS:
        return "Answer the following question by reasoning step-by-step."
    return "Answer the following question."


def format_context(passages: Sequence) -> str:
    """Join passages into the Wikipedia-block format used upstream.

    ``passages`` is a sequence of objects with ``.title`` and ``.text``
    attributes (typically ``src.retrieval.Passage``). Blocks are
    separated by a blank line, matching upstream ``para_to_text`` +
    ``"\\n\\n".join``.
    """
    blocks: list[str] = []
    for p in passages:
        title = (getattr(p, "title", None) or "").strip()
        text = (getattr(p, "text", None) or "").strip()
        if not text:
            continue
        # Mirror para_to_text: if a block already starts with "Wikipedia Title: ", keep it.
        if text.startswith("Wikipedia Title: "):
            blocks.append(text)
        else:
            blocks.append(f"Wikipedia Title: {title}\n{text}")
    return "\n\n".join(blocks)


def build_prompt(
    dataset: str,
    question: str,
    *,
    context_text: str = "",
    generation_so_far: str = "",
) -> str:
    """Assemble the final prompt sent to the LLM.

    Format (mirrors ``StepByStepCOTGenParticipant.query`` upstream)::

        {few_shot}\\n\\n\\n{context_block}Q: {instruction}\\n{question}\\nA: {generation_so_far}

    where ``context_block`` is ``""`` for no-context strategies and
    ``"{context}\\n\\n"`` otherwise.

    ``generation_so_far`` is non-empty only during the IRCoT loop, where
    the model is asked to continue its own partial chain-of-thought.
    """
    with_context = bool(context_text)
    few_shot = load_prompt(dataset, with_context=with_context)

    instruction = _instruction(dataset)
    if context_text:
        test_block = (
            f"{context_text}\n\n"
            f"Q: {instruction}\n{question}\n"
            f"A: {generation_so_far}"
        )
    else:
        test_block = (
            f"Q: {instruction}\n{question}\n"
            f"A: {generation_so_far}"
        )

    return ("\n\n\n".join([few_shot, test_block])).strip()


# ---------------------------------------------------------------------------
# Leakage filter (data_plan.md §2 risk register)
# ---------------------------------------------------------------------------


_METADATA_QID_RE = re.compile(r'^# METADATA:\s*({.*})\s*$', re.MULTILINE)


@lru_cache(maxsize=32)
def _qids_in_file(path_str: str) -> frozenset[str]:
    text = Path(path_str).read_text(encoding="utf8")
    out: set[str] = set()
    for m in _METADATA_QID_RE.finditer(text):
        try:
            meta = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        qid = meta.get("qid")
        if qid:
            out.add(str(qid))
    return frozenset(out)


def leaked_qids(dataset: str) -> frozenset[str]:
    """Union of all qids appearing in this dataset's prompt files.

    We scan *both* the no-context and the gold-with-distractors variants
    because the upstream files often share demonstrations.
    """
    files = [
        prompt_path(dataset, with_context=False),
        prompt_path(dataset, with_context=True),
    ]
    qids: set[str] = set()
    for f in files:
        if f.exists():
            qids |= set(_qids_in_file(str(f)))
    return frozenset(qids)
