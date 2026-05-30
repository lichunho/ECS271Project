"""Answer extraction from raw LLM output.

Ports the regex ``r".* answer is (.*)"`` from
``commaqa/inference/ircot.py::AnswerExtractor.query`` and adds a thin
direct-QA path (first line, trailing-period strip).

Three exports::

    extract_cot(raw)       -> str          # "So the answer is: X." style
    extract_direct(raw)    -> str          # first non-empty line
    has_answer(raw)        -> bool         # IRCoT termination signal

"""

from __future__ import annotations

import re


# Matches Adaptive-RAG's pattern in StepByStepCOTGenParticipant /
# AnswerExtractor.query. We add re.DOTALL so the leading ``.*`` spans
# newlines, since our model output may include several reasoning
# sentences before the "answer is" tail.
_COT_ANSWER_RE = re.compile(r".*answer is\s*:?\s*(.+)", re.IGNORECASE | re.DOTALL)


def _strip_trailing_period(s: str) -> str:
    s = s.strip()
    # Strip a single trailing period (and balanced quotes that wrap the whole answer).
    while s.endswith("."):
        s = s[:-1].rstrip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    return s


def extract_cot(raw: str) -> str:
    """Extract the final answer from a CoT completion.

    Looks for the ``... answer is: X.`` tail. If no match, takes the last
    non-empty line and strips a trailing period (degenerate but matches
    upstream's behaviour with ``match_all_on_failure=True``).
    """
    if not raw:
        return ""
    text = raw.strip()
    m = _COT_ANSWER_RE.search(text)
    if m:
        return _strip_trailing_period(m.group(1))
    # Fallback: last non-empty line, trailing period stripped.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    return _strip_trailing_period(lines[-1])


def extract_direct(raw: str) -> str:
    """Extract the answer from a direct-QA completion.

    The model's continuation after ``A:`` is the answer. Take the first
    non-empty line (the model can hallucinate further Q/A pairs) and strip
    a trailing period.
    """
    if not raw:
        return ""
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Stop at a new turn marker if the model continued with another demo.
        if line.startswith("Q:") or line.startswith("Question:"):
            return ""
        # Drop a leading "A:" if the model echoed it.
        if line.lower().startswith("a:"):
            line = line[2:].strip()
        return _strip_trailing_period(line)
    return ""


def has_answer(raw: str) -> bool:
    """IRCoT termination signal — did the model emit ``... answer is X``?"""
    if not raw:
        return False
    return bool(_COT_ANSWER_RE.search(raw))


def extract(raw: str, *, cot: bool) -> str:
    """Dispatch to the right extractor."""
    return extract_cot(raw) if cot else extract_direct(raw)
