from __future__ import annotations

import pytest

from src.config import LLM_MODEL
from src.llm import answer, get_client, probe


@pytest.fixture(scope="module")
def client():
    try:
        c = get_client()
        available = [m.id for m in c.models.list().data]
    except Exception as e:
        pytest.skip(f"LLM server not reachable: {e}")
    if LLM_MODEL not in available:
        pytest.skip(f"Configured LLM_MODEL={LLM_MODEL!r} not available (server reports: {available})")
    return c


def test_probe_returns_text(client):
    result = probe("What is the capital of France?", client=client)
    assert isinstance(result["text"], str) and result["text"]


def test_probe_returns_mean_logprob(client):
    result = probe("What is the capital of France?", client=client)
    assert isinstance(result["mean_logprob"], float)
    assert len(result["logprobs"]) == len(result["tokens"]) > 0


def test_answer_returns_text(client):
    result = answer("What is the capital of France?", client=client)
    assert isinstance(result["text"], str) and result["text"]
    assert result["mean_logprob"] is None


def test_probe_bounded_by_max_tokens(client):
    from src.config import PROBE_MAX_TOKENS

    result = probe("Describe the water cycle in detail.", client=client)
    assert len(result["tokens"]) <= PROBE_MAX_TOKENS
