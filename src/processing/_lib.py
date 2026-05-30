"""Tiny I/O helpers shared by the processing-script ports.

Mirrors the subset of ``Adaptive-RAG/processing_scripts/lib.py`` we actually
use (``read_json``, ``read_jsonl``, ``write_jsonl``). We deliberately do NOT
port ``find_matching_paragraph_text`` — it requires a running Elasticsearch
retriever which is out of scope for the data-sourcing phase.

Source: https://github.com/starsuzi/Adaptive-RAG/blob/main/processing_scripts/lib.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(file_path: str | Path) -> Any:
    with open(file_path, "r", encoding="utf8", errors="ignore") as fh:
        return json.load(fh)


def read_jsonl(file_path: str | Path) -> list[dict]:
    with open(file_path, "r", encoding="utf8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(instances: list[dict], file_path: str | Path) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf8") as fh:
        for instance in instances:
            fh.write(json.dumps(instance) + "\n")
