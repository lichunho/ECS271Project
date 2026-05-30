"""Three answerer strategies for oracle labelling.

Each strategy answers a question and returns an :class:`Attempt` — a
JSON-serialisable dataclass with the prediction, EM/F1, latency, and
strategy-specific context metadata.

* :func:`no_retrieval` — closed-book; uses the ``no_context_*_flan_t5.txt``
  prompt; **no system prompt** (Adaptive-RAG few-shots are completion-style).
* :func:`single_step` — BM25 top-15 from
  :func:`src.retrieval.get_retriever`, formatted with the
  ``gold_with_1_distractors_context_*_flan_t5.txt`` prompt.
* :func:`multi_step`  — IRCoT loop: BM25 top-``k_per_step``, then while
  the model hasn't emitted ``"So the answer is"`` and we haven't hit the
  iteration / context-token caps, take the new CoT sentence as the next
  query, retrieve more (deduped by ``doc_id``), append, re-prompt.

All three return ``Attempt`` objects, which the orchestrator serialises
into the row's ``attempts`` dict.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from openai import AsyncOpenAI

from src.annotate_lib import extract, llm_adapter, normalise, prompts as prompts_mod
from src.annotate_lib.llm_adapter import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    estimate_tokens,
)

log = logging.getLogger(__name__)

# Soft context cap (proxy: characters ≈ 4 × tokens; 6000 tokens → 24000 chars).
_CONTEXT_CHAR_CAP: int = 24_000

# How many tokens we let the model emit per IRCoT step. Upstream caps each
# sentence at max_length=200 (see base_configs/ircot_qa_flan_t5_xl_*.jsonnet).
_IRCOT_MAX_TOKENS_PER_STEP: int = 200


# ---------------------------------------------------------------------------
# Attempt dataclass
# ---------------------------------------------------------------------------


@dataclass
class Attempt:
    """One strategy's attempt at a question.

    Field set matches data_plan.md §2 (the labeled-row schema). Extra
    strategy-specific fields go in ``context_doc_ids`` / ``n_hops``.
    """
    strategy: str
    pred_raw: str
    pred_extracted: str
    em: int
    f1: float
    ok: bool
    latency_s: float
    prompt_tokens_est: int
    completion_tokens_est: int
    context_doc_ids: list[str] = field(default_factory=list)
    n_hops: int = 0
    error: str = ""

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        # Round floats for cleaner JSON.
        d["f1"] = round(d["f1"], 4)
        d["latency_s"] = round(d["latency_s"], 3)
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Attempt":
        # Backfill missing optional fields for forward-compat with old cache.
        d = dict(d)
        d.setdefault("context_doc_ids", [])
        d.setdefault("n_hops", 0)
        d.setdefault("error", "")
        return cls(**d)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gold_aliases(row: dict) -> list[str]:
    """Pull the gold-answer span list out of the Adaptive-RAG row shape.

    Returns ``[]`` if the row has no usable spans (rare — would mean a
    broken processed dev row).
    """
    out: list[str] = []
    for obj in row.get("answers_objects", []) or []:
        for span in obj.get("spans", []) or []:
            if span:
                out.append(str(span))
        # Some shapes also carry a "number" or "date" — normalize lightly.
        num = obj.get("number")
        if num:
            out.append(str(num))
        date = obj.get("date") or {}
        if isinstance(date, dict):
            parts = [str(date.get(k, "")) for k in ("day", "month", "year")]
            stamp = " ".join(p for p in parts if p)
            if stamp.strip():
                out.append(stamp.strip())
    return out


def _score(pred: str, golds: list[str]) -> tuple[int, float, bool]:
    em_v = normalise.em(pred, golds)
    f1_v = normalise.f1(pred, golds)
    return em_v, f1_v, bool(em_v)


def _truncate_passages_by_chars(passages: list, char_cap: int):
    """Drop passages from the tail until the running text size ≤ ``char_cap``.

    Useful when an IRCoT loop has accumulated many docs.
    """
    out: list = []
    used = 0
    for p in passages:
        approx = len(p.text or "") + len(p.title or "") + 20  # block header overhead
        if used + approx > char_cap and out:
            break
        out.append(p)
        used += approx
    return out


# ---------------------------------------------------------------------------
# no_retrieval
# ---------------------------------------------------------------------------


async def no_retrieval(
    question: str,
    dataset: str,
    *,
    gold_aliases: list[str],
    model: str,
    client: Optional[AsyncOpenAI] = None,
) -> Attempt:
    """Closed-book strategy."""
    prompt = prompts_mod.build_prompt(dataset, question)
    cot = prompts_mod.is_cot(dataset)

    try:
        result = await llm_adapter.complete(
            prompt,
            model=model,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            client=client,
        )
    except Exception as e:
        return Attempt(
            strategy="no_retrieval",
            pred_raw="", pred_extracted="",
            em=0, f1=0.0, ok=False, latency_s=0.0,
            prompt_tokens_est=estimate_tokens(prompt),
            completion_tokens_est=0,
            error=f"{type(e).__name__}: {e}",
        )

    extracted = extract.extract(result.text, cot=cot)
    em_v, f1_v, ok = _score(extracted, gold_aliases)
    pt = result.prompt_tokens if result.prompt_tokens is not None else estimate_tokens(prompt)
    ct = result.completion_tokens if result.completion_tokens is not None else estimate_tokens(result.text)
    return Attempt(
        strategy="no_retrieval",
        pred_raw=result.text,
        pred_extracted=extracted,
        em=em_v, f1=f1_v, ok=ok, latency_s=result.latency_s,
        prompt_tokens_est=pt,
        completion_tokens_est=ct,
    )


# ---------------------------------------------------------------------------
# single_step
# ---------------------------------------------------------------------------


async def single_step(
    question: str,
    dataset: str,
    *,
    gold_aliases: list[str],
    model: str,
    k: int = 15,
    client: Optional[AsyncOpenAI] = None,
    retriever: Any = None,
) -> Attempt:
    """BM25 top-k → prompt with passages → answer."""
    if retriever is None:
        # Lazy import — keeps pyserini's JVM out of test paths that don't need it.
        from src.retrieval import get_retriever
        retriever = get_retriever(dataset)

    try:
        passages = retriever.search(question, k=k)
    except Exception as e:
        return Attempt(
            strategy="single_step",
            pred_raw="", pred_extracted="",
            em=0, f1=0.0, ok=False, latency_s=0.0,
            prompt_tokens_est=0, completion_tokens_est=0,
            error=f"retrieval failed: {type(e).__name__}: {e}",
        )

    passages = _truncate_passages_by_chars(passages, _CONTEXT_CHAR_CAP)
    context = prompts_mod.format_context(passages)
    prompt = prompts_mod.build_prompt(dataset, question, context_text=context)
    cot = prompts_mod.is_cot(dataset)

    try:
        result = await llm_adapter.complete(
            prompt,
            model=model,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            client=client,
        )
    except Exception as e:
        return Attempt(
            strategy="single_step",
            pred_raw="", pred_extracted="",
            em=0, f1=0.0, ok=False, latency_s=0.0,
            prompt_tokens_est=estimate_tokens(prompt),
            completion_tokens_est=0,
            context_doc_ids=[p.doc_id for p in passages],
            error=f"{type(e).__name__}: {e}",
        )

    extracted = extract.extract(result.text, cot=cot)
    em_v, f1_v, ok = _score(extracted, gold_aliases)
    pt = result.prompt_tokens if result.prompt_tokens is not None else estimate_tokens(prompt)
    ct = result.completion_tokens if result.completion_tokens is not None else estimate_tokens(result.text)
    return Attempt(
        strategy="single_step",
        pred_raw=result.text,
        pred_extracted=extracted,
        em=em_v, f1=f1_v, ok=ok, latency_s=result.latency_s,
        prompt_tokens_est=pt,
        completion_tokens_est=ct,
        context_doc_ids=[p.doc_id for p in passages],
    )


# ---------------------------------------------------------------------------
# multi_step (IRCoT)
# ---------------------------------------------------------------------------


# Used to pull the next retrieval query from the LLM's CoT continuation.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _next_query_from_generation(gen_so_far: str, new_chunk: str) -> str:
    """Take the *new* sentence the model just emitted as the next query.

    If the model emitted multiple sentences in one shot we use the last
    non-empty one. Mirrors upstream's per-step "one sentence" cadence
    (Adaptive-RAG's IRCoT generator emits ``max_length=200`` per turn,
    which is typically one to two sentences).
    """
    chunk = new_chunk.strip()
    if not chunk:
        return ""
    parts = [p.strip() for p in _SENT_SPLIT.split(chunk) if p.strip()]
    if not parts:
        return chunk
    return parts[-1]


async def multi_step(
    question: str,
    dataset: str,
    *,
    gold_aliases: list[str],
    model: str,
    max_iters: int = 4,
    k_per_step: int = 6,
    client: Optional[AsyncOpenAI] = None,
    retriever: Any = None,
) -> Attempt:
    """IRCoT loop: retrieve → reason one step → retrieve more → repeat.

    Always treated as CoT (we use the CoT prompt files even for direct-QA
    datasets, since IRCoT only makes sense with a reasoning model).
    """
    if retriever is None:
        from src.retrieval import get_retriever
        retriever = get_retriever(dataset)

    # CoT prompt template — IRCoT is a CoT strategy by definition.
    # For direct-QA datasets we still use the CoT prompt for multi_step
    # because the loop needs the "So the answer is" termination signal.
    # We monkey-patch the dataset hint into a CoT alias below.
    cot_dataset = dataset if dataset in prompts_mod.COT_DATASETS else "hotpotqa"

    accumulated: list = []  # list of Passage objects
    seen_doc_ids: set[str] = set()
    generation_so_far = ""
    total_latency = 0.0
    total_pt = 0
    total_ct = 0
    last_raw = ""
    n_hops = 0
    error = ""

    # First retrieval: use the user's question.
    try:
        hits = retriever.search(question, k=k_per_step)
    except Exception as e:
        return Attempt(
            strategy="multi_step",
            pred_raw="", pred_extracted="",
            em=0, f1=0.0, ok=False, latency_s=0.0,
            prompt_tokens_est=0, completion_tokens_est=0,
            error=f"retrieval failed: {type(e).__name__}: {e}",
        )
    for p in hits:
        if p.doc_id not in seen_doc_ids:
            accumulated.append(p)
            seen_doc_ids.add(p.doc_id)

    for hop in range(max_iters):
        n_hops = hop + 1
        passages = _truncate_passages_by_chars(accumulated, _CONTEXT_CHAR_CAP)
        context = prompts_mod.format_context(passages)
        prompt = prompts_mod.build_prompt(
            cot_dataset, question,
            context_text=context,
            generation_so_far=generation_so_far,
        )
        try:
            result = await llm_adapter.complete(
                prompt,
                model=model,
                max_tokens=_IRCOT_MAX_TOKENS_PER_STEP,
                temperature=DEFAULT_TEMPERATURE,
                client=client,
            )
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            break
        last_raw = result.text
        total_latency += result.latency_s
        total_pt += result.prompt_tokens if result.prompt_tokens is not None else estimate_tokens(prompt)
        total_ct += result.completion_tokens if result.completion_tokens is not None else estimate_tokens(result.text)

        new_chunk = result.text.strip()
        # Update the running CoT generation.
        if generation_so_far and not generation_so_far.endswith(" "):
            generation_so_far = generation_so_far + " " + new_chunk
        else:
            generation_so_far = generation_so_far + new_chunk

        # Termination: model emitted "... answer is ...".
        if extract.has_answer(generation_so_far):
            break

        # Otherwise: feed the latest sentence back as the next query.
        next_q = _next_query_from_generation(generation_so_far, new_chunk)
        if not next_q:
            break  # model gave up
        try:
            new_hits = retriever.search(next_q, k=k_per_step)
        except Exception as e:
            error = f"retrieval failed mid-loop: {type(e).__name__}: {e}"
            break
        added = 0
        for p in new_hits:
            if p.doc_id not in seen_doc_ids:
                accumulated.append(p)
                seen_doc_ids.add(p.doc_id)
                added += 1
        if added == 0:
            # No new docs — further hops won't help; bail.
            break
        # If accumulated context is past the cap *and* we already emitted
        # everything we can, stop.
        running_chars = sum(len(p.text or "") + 20 for p in accumulated)
        if running_chars >= _CONTEXT_CHAR_CAP and hop + 1 < max_iters:
            log.debug("IRCoT context cap hit after %d hops — stopping", n_hops)
            break

    # Extract final answer from whatever we've got (CoT-style).
    extracted = extract.extract_cot(generation_so_far or last_raw)
    em_v, f1_v, ok = _score(extracted, gold_aliases)
    return Attempt(
        strategy="multi_step",
        pred_raw=generation_so_far or last_raw,
        pred_extracted=extracted,
        em=em_v, f1=f1_v, ok=ok, latency_s=total_latency,
        prompt_tokens_est=total_pt,
        completion_tokens_est=total_ct,
        context_doc_ids=[p.doc_id for p in accumulated],
        n_hops=n_hops,
        error=error,
    )
