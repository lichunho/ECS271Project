"""Offline unit tests for the annotation library.

These tests mock out the LLM and retriever so they run without a JDK or
a running LLM server. They cover:

* SQuAD ``normalize_answer`` / EM / F1
* CoT and direct answer extraction
* Prompt assembly + leakage-qid scanning
* Cheapest-wins label assignment + binary fallback
* SQLite cache round-trip
* End-to-end ``no_retrieval`` pipeline with a monkeypatched LLM
* End-to-end ``single_step`` pipeline with a fake retriever
* ``multi_step`` IRCoT termination on "answer is"

Run with::

    .\\.venv\\Scripts\\python.exe -m pytest tests\\test_annotate_strategies.py -q
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

# Ensure the repo root is on sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

from src.annotate_lib import cache as cache_mod
from src.annotate_lib import extract, label, llm_adapter, normalise
from src.annotate_lib import prompts as prompts_mod
from src.annotate_lib import strategies as strat_mod


# ---------------------------------------------------------------------------
# normalise.py
# ---------------------------------------------------------------------------


def test_normalize_answer_basics():
    assert normalise.normalize_answer("The Mona Lisa.") == "mona lisa"
    assert normalise.normalize_answer("AN APPLE") == "apple"
    assert normalise.normalize_answer("  a  cat  ") == "cat"
    assert normalise.normalize_answer("") == ""


def test_em_and_f1_over_aliases():
    assert normalise.em("Apple", ["apple", "banana"]) == 1
    assert normalise.em("the apple", ["apple"]) == 1
    assert normalise.em("orange", ["apple"]) == 0
    assert normalise.f1("the apple pie", ["apple pie"]) == 1.0
    assert 0.0 < normalise.f1("apple", ["apple pie"]) < 1.0
    assert normalise.f1("xxxx", ["apple"]) == 0.0


# ---------------------------------------------------------------------------
# extract.py
# ---------------------------------------------------------------------------


def test_extract_cot_pulls_after_answer_is():
    raw = ("The actor was Scott Glenn. So the answer is: Scott Glenn.")
    assert extract.extract_cot(raw) == "Scott Glenn"


def test_extract_cot_handles_multiline_and_dotall():
    raw = "Step 1.\nStep 2.\nSo the answer is: Walls and Bridges."
    assert extract.extract_cot(raw) == "Walls and Bridges"


def test_extract_cot_fallback_last_line():
    raw = "I don't know but maybe\nGermany"
    assert extract.extract_cot(raw) == "Germany"


def test_extract_direct_first_line():
    assert extract.extract_direct(" Walls and Bridges.\nfoo") == "Walls and Bridges"


def test_extract_direct_strips_echoed_A_prefix():
    assert extract.extract_direct("A: Walls and Bridges.") == "Walls and Bridges"


def test_has_answer():
    assert extract.has_answer("Step 1. So the answer is: X.")
    assert not extract.has_answer("Step 1. Still thinking.")


# ---------------------------------------------------------------------------
# prompts.py
# ---------------------------------------------------------------------------


def test_load_prompt_files_exist():
    for ds in prompts_mod.ALL_DATASETS:
        with_context_path = prompts_mod.prompt_path(ds, with_context=True)
        no_context_path = prompts_mod.prompt_path(ds, with_context=False)
        assert with_context_path.exists(), f"missing {with_context_path}"
        assert no_context_path.exists(), f"missing {no_context_path}"


def test_build_prompt_no_context_format():
    p = prompts_mod.build_prompt("musique", "Who wrote X?")
    # Few-shot block, then triple newline, then test example.
    assert "\n\n\n" in p
    assert p.endswith("Q: Answer the following question by reasoning step-by-step.\nWho wrote X?\nA:")


def test_build_prompt_direct_uses_direct_instruction():
    p = prompts_mod.build_prompt("nq", "Who wrote X?")
    assert p.endswith("Q: Answer the following question.\nWho wrote X?\nA:")


def test_build_prompt_with_context_uses_wiki_blocks():
    passages = [SimpleNamespace(title="Foo", text="foo bar.", doc_id="d1", score=1.0)]
    ctx = prompts_mod.format_context(passages)
    assert ctx == "Wikipedia Title: Foo\nfoo bar."
    p = prompts_mod.build_prompt("hotpotqa", "What?", context_text=ctx)
    assert "Wikipedia Title: Foo" in p


def test_leaked_qids_returns_nonempty_set():
    # The HotpotQA prompts contain ~50 unique qids across both variants.
    qids = prompts_mod.leaked_qids("hotpotqa")
    assert len(qids) > 5
    # Sample one I saw during file inspection.
    assert "5a8ed9f355429917b4a5bddd" in qids


# ---------------------------------------------------------------------------
# label.py
# ---------------------------------------------------------------------------


def test_label_cheapest_wins_no_retrieval():
    attempts = {
        "no_retrieval": {"ok": True},
        "single_step":  {"ok": True},
        "multi_step":   {"ok": True},
    }
    assert label.assign_label(attempts=attempts, source_dataset="hotpotqa") == \
        ("no_retrieval", "silver")


def test_label_single_step_beats_multi():
    attempts = {
        "no_retrieval": {"ok": False},
        "single_step":  {"ok": True},
        "multi_step":   {"ok": True},
    }
    assert label.assign_label(attempts=attempts, source_dataset="musique") == \
        ("single_step", "silver")


def test_label_only_multi_passes():
    attempts = {
        "no_retrieval": {"ok": False},
        "single_step":  {"ok": False},
        "multi_step":   {"ok": True},
    }
    assert label.assign_label(attempts=attempts, source_dataset="musique") == \
        ("multi_step", "silver")


def test_label_binary_fallback_multi_hop():
    attempts = {s: {"ok": False} for s in ("no_retrieval", "single_step", "multi_step")}
    assert label.assign_label(attempts=attempts, source_dataset="musique") == \
        ("multi_step", "binary_fallback")


def test_label_binary_fallback_single_hop():
    attempts = {s: {"ok": False} for s in ("no_retrieval", "single_step", "multi_step")}
    assert label.assign_label(attempts=attempts, source_dataset="nq") == \
        ("single_step", "binary_fallback")


def test_label_handles_missing_strategy_keys():
    # User ran with --strategies no_retrieval only
    attempts = {"no_retrieval": {"ok": True}}
    assert label.assign_label(attempts=attempts, source_dataset="squad") == \
        ("no_retrieval", "silver")


# ---------------------------------------------------------------------------
# cache.py
# ---------------------------------------------------------------------------


def test_cache_roundtrip(tmp_path):
    c = cache_mod.AttemptCache(tmp_path / "cache.sqlite")
    assert c.get("h1", "no_retrieval", "m1", "p1") is None
    assert c.n_misses == 1
    payload = {"strategy": "no_retrieval", "ok": True, "em": 1, "f1": 1.0}
    c.put("h1", "no_retrieval", "m1", "p1", payload, created_at=time.time())
    got = c.get("h1", "no_retrieval", "m1", "p1")
    assert got == payload
    assert c.n_hits == 1
    # Different model is a different key.
    assert c.get("h1", "no_retrieval", "m2", "p1") is None
    c.close()


def test_question_hash_stable():
    h1 = cache_mod.question_hash(" What is X? ")
    h2 = cache_mod.question_hash("what is x?")
    assert h1 == h2  # strip + lower normalised
    assert len(h1) == 40  # sha1 hex


# ---------------------------------------------------------------------------
# llm_adapter mocking helpers
# ---------------------------------------------------------------------------


def _fake_ollama_payload(text: str) -> dict:
    """Mirror the shape Ollama's /api/chat returns (the bits we use)."""
    return {
        "message": {"role": "assistant", "content": text},
        "prompt_eval_count": 100,
        "eval_count": 12,
    }


class _FakeAsyncClient:
    """Stand-in for httpx.AsyncClient — emits a fixed text per /api/chat POST."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls: list[str] = []

    async def post(self, url, *, json):  # noqa: A002 — match httpx signature
        # ``json`` is the request body dict. The prompt is the lone user
        # message; record it so tests can assert on prompt assembly.
        self.calls.append(json["messages"][0]["content"])
        text = self._replies.pop(0) if self._replies else ""

        async def _aread():
            return b""

        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: _fake_ollama_payload(text),
            status_code=200,
            text="",
            aread=_aread,
        )


# ---------------------------------------------------------------------------
# Strategy: no_retrieval end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_retrieval_direct_dataset_em(monkeypatch):
    fake = _FakeAsyncClient(replies=["Walls and Bridges."])
    monkeypatch.setattr(llm_adapter, "get_async_client", lambda timeout_s=60.0: fake)

    att = await strat_mod.no_retrieval(
        "What album?", "nq",
        gold_aliases=["Walls and Bridges"], model="fake-model", client=fake,
    )
    assert att.strategy == "no_retrieval"
    assert att.pred_raw == "Walls and Bridges."
    assert att.pred_extracted == "Walls and Bridges"
    assert att.em == 1 and att.ok and att.f1 == 1.0
    assert att.latency_s >= 0
    # The few-shot prompt should be present in the first (and only) call.
    assert fake.calls and "Q: Answer the following question." in fake.calls[0]


@pytest.mark.asyncio
async def test_no_retrieval_cot_dataset_em(monkeypatch):
    fake = _FakeAsyncClient(replies=[
        "Step 1. Bob did X. Step 2. So the answer is: Bob."
    ])
    att = await strat_mod.no_retrieval(
        "Who did X?", "musique",
        gold_aliases=["Bob"], model="fake-model", client=fake,
    )
    assert att.pred_extracted == "Bob"
    assert att.em == 1 and att.ok
    # CoT prompt has the "step-by-step" instruction.
    assert "Q: Answer the following question by reasoning step-by-step." in fake.calls[0]


@pytest.mark.asyncio
async def test_no_retrieval_em_fail():
    fake = _FakeAsyncClient(replies=["Wrong."])
    att = await strat_mod.no_retrieval(
        "What?", "nq", gold_aliases=["Right"], model="fake-model", client=fake,
    )
    assert att.em == 0 and not att.ok
    # F1 should be 0 for entirely-disjoint single-word strings.
    assert att.f1 == 0.0


# ---------------------------------------------------------------------------
# Strategy: single_step with fake retriever
# ---------------------------------------------------------------------------


class _FakeRetriever:
    def __init__(self, hits):
        self._hits = hits
        self.queries: list[str] = []

    def search(self, query, k=15, allowed_titles=None):
        self.queries.append(query)
        return list(self._hits[:k])


@pytest.mark.asyncio
async def test_single_step_passes_context_into_prompt():
    fake_llm = _FakeAsyncClient(replies=["Apple Records"])
    hits = [
        SimpleNamespace(title="Walls and Bridges", text="Apple Records released the album.",
                        doc_id="d1", score=2.5),
        SimpleNamespace(title="Nobody Loves You", text="A John Lennon song.",
                        doc_id="d2", score=1.5),
    ]
    fake_ret = _FakeRetriever(hits)

    att = await strat_mod.single_step(
        "Who released the album?", "nq",
        gold_aliases=["Apple Records"], model="fake-model",
        k=2, client=fake_llm, retriever=fake_ret,
    )
    assert att.strategy == "single_step"
    assert att.em == 1 and att.ok
    assert att.context_doc_ids == ["d1", "d2"]
    # The wiki block should be in the prompt.
    assert "Wikipedia Title: Walls and Bridges" in fake_llm.calls[0]
    # And the retriever was queried with the user question.
    assert fake_ret.queries == ["Who released the album?"]


# ---------------------------------------------------------------------------
# Strategy: multi_step IRCoT termination
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_step_terminates_on_answer_is():
    fake_llm = _FakeAsyncClient(replies=[
        "Step 1. The album is Walls and Bridges. So the answer is: Walls and Bridges.",
    ])
    hits = [SimpleNamespace(title="Walls and Bridges", text="...",
                            doc_id="d1", score=1.0)]
    fake_ret = _FakeRetriever(hits)

    att = await strat_mod.multi_step(
        "Album?", "hotpotqa",
        gold_aliases=["Walls and Bridges"], model="fake-model",
        max_iters=4, k_per_step=6,
        client=fake_llm, retriever=fake_ret,
    )
    assert att.strategy == "multi_step"
    assert att.em == 1 and att.ok
    assert att.n_hops == 1  # terminated on first hop
    assert att.context_doc_ids == ["d1"]


class _UniqueHitsRetriever:
    """Returns a fresh, unique doc per query — exercises the IRCoT loop fully."""
    def __init__(self):
        self.queries: list[str] = []
        self._counter = 0

    def search(self, query, k=15, allowed_titles=None):
        self.queries.append(query)
        out = []
        for _ in range(k):
            i = self._counter
            self._counter += 1
            out.append(SimpleNamespace(
                title=f"T{i}", text=f"text{i}", doc_id=f"d{i}", score=1.0,
            ))
        return out


@pytest.mark.asyncio
async def test_multi_step_iterates_until_max():
    # Model never emits "answer is" — should hit max_iters.
    fake_llm = _FakeAsyncClient(replies=[
        "Step 1. thinking.",
        "Step 2. still thinking.",
        "Step 3. hmm.",
        "Step 4. give up.",
    ])
    fake_ret = _UniqueHitsRetriever()

    att = await strat_mod.multi_step(
        "what?", "hotpotqa",
        gold_aliases=["xyz"], model="fake-model",
        max_iters=4, k_per_step=2,
        client=fake_llm, retriever=fake_ret,
    )
    assert att.n_hops == 4
    assert att.ok is False


# ---------------------------------------------------------------------------
# Attempt JSON round-trip
# ---------------------------------------------------------------------------


def test_attempt_json_roundtrip():
    a = strat_mod.Attempt(
        strategy="no_retrieval", pred_raw="x", pred_extracted="x",
        em=1, f1=1.0, ok=True, latency_s=0.1,
        prompt_tokens_est=100, completion_tokens_est=5,
    )
    d = a.to_json()
    assert json.loads(json.dumps(d))  # serialisable
    b = strat_mod.Attempt.from_json(d)
    assert b.ok and b.em == 1


# ---------------------------------------------------------------------------
# annotate_one_row: partial-rerun behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_annotate_one_row_picks_up_cached_strategies_outside_subset(tmp_path):
    """Partial re-run with --strategies single_step should still emit any
    cached no_retrieval / multi_step attempts in the row, so the output file
    isn't impoverished and cheapest-wins labelling sees the full picture.
    """
    # Load the orchestrator module. scripts/ is a package, so this works.
    from scripts import annotate as ann

    # Seed the cache with a passing no_retrieval payload for the question.
    cache = cache_mod.AttemptCache(tmp_path / "cache.sqlite")
    question = "Who released Walls and Bridges?"
    qh = cache_mod.question_hash(question)
    no_retr_payload = strat_mod.Attempt(
        strategy="no_retrieval",
        pred_raw="Apple Records.", pred_extracted="Apple Records",
        em=1, f1=1.0, ok=True, latency_s=0.1,
        prompt_tokens_est=80, completion_tokens_est=3,
    ).to_json()
    cache.put(
        qh, "no_retrieval", "fake-model", prompts_mod.PROMPT_SET_ID,
        no_retr_payload, created_at=time.time(),
    )

    # single_step will be the only strategy actually invoked.
    fake_llm = _FakeAsyncClient(replies=["Apple Records"])
    fake_ret = _FakeRetriever(hits=[
        SimpleNamespace(title="Walls and Bridges",
                        text="Apple Records released the album.",
                        doc_id="d1", score=2.5),
    ])

    row = {
        "question_id": "q1",
        "question_text": question,
        "answers_objects": [{"spans": ["Apple Records"]}],
    }

    out = await ann.annotate_one_row(
        row,
        source_dataset="nq",
        strategies=("single_step",),
        model="fake-model",
        bm25_k=1, ircot_max_iters=1, ircot_k_per_step=1,
        cache=cache, client=fake_llm, retriever=fake_ret,
        semaphore=asyncio.Semaphore(1),
    )

    # Output row carries both the freshly-run and the back-filled attempt.
    assert set(out["attempts"].keys()) == {"no_retrieval", "single_step"}
    assert out["attempts"]["no_retrieval"] == no_retr_payload
    assert out["attempts"]["single_step"]["ok"] is True
    # And cheapest-wins picks the cached no_retrieval — not single_step.
    assert out["oracle_label"] == "no_retrieval"
    assert out["label_source"] == "silver"

    # multi_step wasn't cached, so it should simply be absent (not faked-fail).
    assert "multi_step" not in out["attempts"]

    cache.close()
