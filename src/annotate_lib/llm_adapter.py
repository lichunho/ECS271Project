"""Bare-user-role LLM client for completion-style few-shot prompts.

A naive chat client would inject a "You are a concise QA assistant"
system prompt — fine for an interactive probe/answer path, but it would
poison the Adaptive-RAG few-shot prompts, which are designed to be
continued literally. This module sidesteps that by sending a single
``{"role": "user"}`` message containing the *entire* assembled prompt.

**Why native Ollama endpoint, not OpenAI-compat:**
The few-shot labelling path is fundamentally incompatible with reasoning
models that emit a chain-of-thought before the final answer. Modern
locally-hosted instruct models (gemma4, qwen3, deepseek-r1, ...) are all
"thinking" models and ship that mode on by default. Ollama exposes a
``think`` flag in its native ``/api/chat`` that turns thinking off and
makes the model behave like a traditional instruct model — but Ollama's
OpenAI-compatibility layer (``/v1/chat/completions``) silently drops the
``think`` field, leaving thinking on. We therefore talk to ``/api/chat``
directly. This also keeps the prompt path 100% identical to ``ollama run``
behaviour, which is what we test against.

If the user ever points ``LLM_BASE_URL`` at a non-Ollama backend
(vLLM, hosted OpenAI/Anthropic, …), this module needs to grow a branch.
For now we assume Ollama; the smoke test in the orchestrator catches a
wrong URL immediately.

Key parameters (data_plan.md §2):
    temperature=0, max_tokens=200, greedy (no top_p / top_k).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from src.config import LLM_BASE_URL, LLM_MODEL

log = logging.getLogger(__name__)


# Decoding constants — match data_plan.md §2 ("temperature=0, max_tokens=200").
DEFAULT_MAX_TOKENS: int = 200
DEFAULT_TEMPERATURE: float = 0.0

# Per-request timeout (seconds). Local Ollama can stall on the first request
# while warming up the model; subsequent calls are fast.
DEFAULT_TIMEOUT_S: float = 60.0
DEFAULT_RETRIES: int = 3
DEFAULT_BACKOFF_S: float = 2.0

# Disable thinking-mode on models that support it (gemma4, qwen3, deepseek-r1,
# etc.). Few-shot completion prompts expect the model to continue with a bare
# answer ("A: ...") — if the model thinks first, it either returns the
# reasoning trace as content (failing EM) or routes the answer to a separate
# ``reasoning`` channel. Disabling thinking bypasses both failure modes.
# Ollama silently ignores the field for models that don't expose a thinking
# toggle, so this is safe to send unconditionally.
DEFAULT_THINK: bool = False


@dataclass
class CompletionResult:
    text: str
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None


# ---------------------------------------------------------------------------
# Ollama base-URL handling
# ---------------------------------------------------------------------------
#
# ``LLM_BASE_URL`` is the OpenAI-compat URL (``http://host:11434/v1``).
# Ollama's native chat endpoint lives at ``http://host:11434/api/chat``. We
# strip ``/v1`` if present so callers can leave their config unchanged.


def _ollama_base() -> str:
    base = LLM_BASE_URL.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


def _chat_url() -> str:
    return _ollama_base() + "/api/chat"


_ASYNC_CLIENT_SINGLETON: httpx.AsyncClient | None = None


def get_async_client(timeout_s: float = DEFAULT_TIMEOUT_S) -> httpx.AsyncClient:
    """Cache one httpx.AsyncClient per process.

    Shares HTTP connection pooling across all annotation calls.
    """
    global _ASYNC_CLIENT_SINGLETON
    if _ASYNC_CLIENT_SINGLETON is None:
        _ASYNC_CLIENT_SINGLETON = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        )
    return _ASYNC_CLIENT_SINGLETON


async def complete(
    prompt: str,
    *,
    model: str = LLM_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    client: Optional[httpx.AsyncClient] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
    backoff_s: float = DEFAULT_BACKOFF_S,
    stop: Optional[list[str]] = None,
    think: bool = DEFAULT_THINK,
) -> CompletionResult:
    """Send a bare user-role completion request and return the raw text.

    No system prompt is injected — the entire ``prompt`` string is the
    sole user-role message. This is the format Adaptive-RAG's few-shot
    files expect.

    ``think`` (default False) toggles Ollama's thinking-mode flag for
    models that support it (gemma4, qwen3, deepseek-r1, etc.). We always
    want it off for the few-shot labelling path. For models without a
    thinking toggle Ollama silently ignores it.

    Retries on connection/timeout errors up to ``retries`` times with
    exponential backoff.
    """
    client = client or get_async_client(timeout_s=timeout_s)
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "think": think,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if stop:
        body["options"]["stop"] = stop

    last_err: Exception | None = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            resp = await client.post(_chat_url(), json=body)
            resp.raise_for_status()
            payload = resp.json()
            elapsed = time.time() - t0

            message = payload.get("message") or {}
            text = message.get("content") or ""
            # Defensive: if thinking was somehow re-enabled and the answer
            # lives in ``thinking``, fall back to that so the extractor has
            # something to chew on.
            if not text:
                text = message.get("thinking") or ""

            return CompletionResult(
                text=text,
                latency_s=elapsed,
                prompt_tokens=payload.get("prompt_eval_count"),
                completion_tokens=payload.get("eval_count"),
            )
        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as e:
            last_err = e
            wait = backoff_s * (2 ** attempt)
            log.warning(
                "LLM call %s (attempt %d/%d) — retrying in %.1fs",
                type(e).__name__, attempt + 1, retries, wait,
            )
            await asyncio.sleep(wait)
        except httpx.HTTPStatusError as e:
            # 4xx is a bug in our request — don't retry blindly.
            if 400 <= e.response.status_code < 500:
                raise RuntimeError(
                    f"LLM endpoint returned {e.response.status_code}: "
                    f"{e.response.text[:500]}"
                ) from e
            last_err = e
            wait = backoff_s * (2 ** attempt)
            log.warning(
                "LLM call HTTP %d (attempt %d/%d) — retrying in %.1fs",
                e.response.status_code, attempt + 1, retries, wait,
            )
            await asyncio.sleep(wait)
        except Exception as e:  # pragma: no cover — unexpected wire error
            last_err = e
            log.exception("LLM call failed unexpectedly (attempt %d/%d): %s",
                          attempt + 1, retries, e)
            await asyncio.sleep(backoff_s * (2 ** attempt))
    raise RuntimeError(
        f"LLM call exhausted {retries} retries; last error: {last_err}"
    )


def estimate_tokens(s: str) -> int:
    """Cheap proxy when usage counts are unavailable: ~4 chars / token."""
    if not s:
        return 0
    return max(1, len(s) // 4)
