"""Supporting modules for ``scripts/annotate.py`` — see ``data_plan.md`` §2.

Public surface (imported on demand by ``annotate.py``):

* :mod:`src.annotate_lib.normalise` — SQuAD ``normalize_answer`` + EM/F1.
* :mod:`src.annotate_lib.prompts`   — load + compose vendored few-shot files
  under ``prompts/{dataset}/``.
* :mod:`src.annotate_lib.extract`   — ``.* answer is (.*)`` regex extractor.
* :mod:`src.annotate_lib.llm_adapter` — bare-user-role async LLM client.
* :mod:`src.annotate_lib.strategies` — ``no_retrieval`` / ``single_step`` /
  ``multi_step`` (IRCoT).
* :mod:`src.annotate_lib.cache`     — SQLite WAL cache keyed on
  ``(question_hash, strategy, model_id, prompt_set_id)``.
* :mod:`src.annotate_lib.label`     — cheapest-wins assignment + binary
  fallback per :func:`label_complexity` upstream.
"""
