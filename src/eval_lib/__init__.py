"""Offline evaluation harness for the Adaptive-RAG router.

Because the labeller and answerer are the same model, every strategy's metrics
are already stored per question in ``data/labeled/{split}/*.jsonl``. Evaluation
is therefore a deterministic *offline join*: map each question to a chosen route,
look up ``attempts[chosen]``, aggregate. This package reads only JSONL + a tau
grid; it never imports the LLM adapter and makes no network/model calls.
"""
