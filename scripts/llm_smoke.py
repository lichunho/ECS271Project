from __future__ import annotations

from src.config import LLM_BASE_URL, LLM_MODEL
from src.llm import answer, probe


def main() -> None:
    question = "What is the capital of France?"

    print(f"Endpoint: {LLM_BASE_URL}")
    print(f"Model:    {LLM_MODEL}")
    print(f"Question: {question}\n")

    p = probe(question)
    print(f"Probe text:     {p['text']!r}")
    print(f"Probe tokens:   {len(p['tokens'])}")
    if p["mean_logprob"] is not None:
        print(f"Mean logprob:   {p['mean_logprob']:.4f}")
    else:
        print("Mean logprob:   (no logprobs returned)")

    a = answer(question)
    print(f"\nAnswer text:    {a['text']!r}")


if __name__ == "__main__":
    main()
