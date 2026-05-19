from __future__ import annotations

from openai import OpenAI

from src.config import (
    ANSWER_MAX_TOKENS,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    PROBE_MAX_TOKENS,
)

_SYSTEM_PROMPT = (
    "You are a concise question-answering assistant. Answer the user's question directly."
)


def get_client() -> OpenAI:
    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


def _build_messages(question: str, context: str | None) -> list[dict]:
    if context:
        user_content = f"Context:\n{context}\n\nQuestion: {question}"
    else:
        user_content = question
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _complete(
    client: OpenAI,
    messages: list[dict],
    max_tokens: int,
    want_logprobs: bool,
) -> dict:
    kwargs = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": LLM_TEMPERATURE,
    }
    if want_logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 1

    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    text = choice.message.content or ""

    tokens: list[str] = []
    token_logprobs: list[float] = []
    mean_logprob: float | None = None

    if want_logprobs and choice.logprobs and choice.logprobs.content:
        for entry in choice.logprobs.content:
            tokens.append(entry.token)
            token_logprobs.append(entry.logprob)
        if token_logprobs:
            mean_logprob = sum(token_logprobs) / len(token_logprobs)

    return {
        "text": text,
        "tokens": tokens,
        "logprobs": token_logprobs,
        "mean_logprob": mean_logprob,
    }


def probe(
    question: str,
    context: str | None = None,
    client: OpenAI | None = None,
) -> dict:
    client = client or get_client()
    messages = _build_messages(question, context)
    return _complete(client, messages, PROBE_MAX_TOKENS, want_logprobs=True)


def answer(
    question: str,
    context: str | None = None,
    client: OpenAI | None = None,
) -> dict:
    client = client or get_client()
    messages = _build_messages(question, context)
    return _complete(client, messages, ANSWER_MAX_TOKENS, want_logprobs=False)
