"""Ports of Adaptive-RAG's processing scripts.

Source: https://github.com/starsuzi/Adaptive-RAG/tree/main/processing_scripts

Each module mirrors one upstream file. We strip absolute path assumptions
(originals hard-coded ``processed_data/...``) and expose a ``main(input_dir,
output_dir, ...)`` function so the orchestrator can call them directly. The
processing logic itself is preserved verbatim, including ``random.seed(13370)``
and ``sample_size = 500``.
"""

from __future__ import annotations
