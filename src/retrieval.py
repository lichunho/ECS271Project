"""BM25 retrieval client for the Adaptive-RAG reproduction.

Provides a small, stable API consumed by the annotation step
(``scripts/annotate.py`` — pending) and by the paragraph-remap pass in
``src/processing/subsample.py``.

Design notes
------------
- We use Pyserini's ``LuceneSearcher`` over local Lucene indices instead of
  the Elasticsearch HTTP service used by upstream Adaptive-RAG
  (`retriever_server/serve.py`). The query semantics we expose
  (``query``, ``k``, ``allowed_titles``) mirror Adaptive-RAG's
  ``retrieve_from_elasticsearch`` contract — see
  ``ElasticsearchRetriever.retrieve_paragraphs`` in their repo.
- For NQ/Trivia/SQuAD we share Pyserini's prebuilt ``wikipedia-dpr-100w``
  index. For HotpotQA/2Wiki/MuSiQue we build per-dataset indices from each
  dataset's bundled corpus (see ``scripts/build_retrieval.py``).
- BM25 parameters: ``k1=0.9, b=0.4``. These are Anserini/Pyserini defaults
  and what Pyserini's ``LuceneSearcher.set_bm25`` uses if no arguments are
  passed. Adaptive-RAG's `serve.py` lets Elasticsearch use *its* defaults
  (``k1=1.2, b=0.75``); we accept the small scoring divergence as documented
  in ``data_plan.md`` ("slight scoring divergence from paper's ES BM25 —
  documented as a deliberate trade").

Each ``Passage`` has the same field names as the Adaptive-RAG ES document
``_source`` so downstream code can be schema-agnostic.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)

# Datasets that share the prebuilt wiki BM25 index (DPR's 100-word splits).
_WIKI_DATASETS: tuple[str, ...] = ("nq", "trivia", "squad", "wiki")

# Datasets that have their own per-dataset BM25 index built from the bundled corpus.
_PER_DATASET_INDICES: tuple[str, ...] = ("hotpotqa", "2wikimultihopqa", "musique")

# Pyserini's prebuilt-index name for the DPR 100-word Wikipedia split.
PREBUILT_WIKI_INDEX = "wikipedia-dpr-100w"

# Default on-disk root for per-dataset Lucene indices, mirroring the layout
# in ``data_plan.md`` (``data/indices/pyserini/{dataset}``). Override via
# the ``ECS271_INDEX_ROOT`` env var if your project root is elsewhere.
DEFAULT_INDEX_ROOT = Path(
    os.environ.get(
        "ECS271_INDEX_ROOT",
        str(Path(__file__).resolve().parent.parent / "data" / "indices" / "pyserini"),
    )
)


@dataclass(frozen=True)
class Passage:
    """A single BM25 hit.

    Field names match Adaptive-RAG's ES ``_source`` schema so that
    downstream code that previously consumed their retriever's JSON works
    unchanged.
    """

    doc_id: str
    title: str
    text: str  # the paragraph_text
    score: float


class BM25Retriever:
    """Thin wrapper over Pyserini's ``LuceneSearcher`` matching the
    upstream ``retrieve_from_elasticsearch`` contract.

    Parameters
    ----------
    index_dir : Path
        Either a local Lucene index directory OR the literal string
        ``"prebuilt:wikipedia-dpr-100w"`` (or any other Pyserini prebuilt
        index name prefixed with ``prebuilt:``). Use ``get_retriever()``
        below to pick the right one for a dataset.
    k1, b : float
        BM25 hyperparameters. Defaults match Pyserini's ``set_bm25()``
        defaults (``k1=0.9, b=0.4``), which are also Anserini's defaults.
    """

    def __init__(self, index_dir: Path | str, k1: float = 0.9, b: float = 0.4) -> None:
        # Local import: Pyserini boots a JVM on import — keep that out of
        # module-level so test runners and other consumers that don't need
        # retrieval (e.g. data-sourcing tests) can ``import src.retrieval``
        # without a JDK present.
        from pyserini.search.lucene import LuceneSearcher

        self.index_dir = str(index_dir)
        if isinstance(index_dir, str) and index_dir.startswith("prebuilt:"):
            prebuilt_name = index_dir.split(":", 1)[1]
            log.info("Loading Pyserini prebuilt index: %s", prebuilt_name)
            self._searcher = LuceneSearcher.from_prebuilt_index(prebuilt_name)
            if self._searcher is None:
                raise RuntimeError(
                    f"Failed to load prebuilt Pyserini index: {prebuilt_name!r}. "
                    "Either the name is wrong or the download failed."
                )
        else:
            local_path = Path(index_dir)
            if not local_path.exists():
                raise FileNotFoundError(
                    f"Lucene index directory does not exist: {local_path}. "
                    "Run scripts/build_retrieval.py first."
                )
            self._searcher = LuceneSearcher(str(local_path))

        self._searcher.set_bm25(k1, b)
        self.k1 = k1
        self.b = b

    def search(
        self,
        query: str,
        k: int = 15,
        allowed_titles: list[str] | None = None,
    ) -> list[Passage]:
        """Return the top-``k`` passages for ``query``.

        Mirrors ``ElasticsearchRetriever.retrieve_paragraphs``:
          - searches the ``paragraph_text`` field by default,
          - if ``allowed_titles`` is given, post-filters hits to keep only
            those whose ``title`` (case-insensitive, stripped) matches.

        Notes:
          - We over-fetch (``k * 3`` or ``100``, whichever is larger) when
            ``allowed_titles`` is set, then filter and re-truncate to ``k``.
            This matches upstream behaviour: a ``max_buffer_count=100`` is
            fetched, deduped by lowercased ``paragraph_text``, then filtered
            by title.
          - We also dedupe by lowercased text to mirror upstream.
        """
        if not query:
            return []

        fetch_k = max(k * 3, 100) if allowed_titles else k
        try:
            hits = self._searcher.search(query, k=fetch_k)
        except Exception as e:
            # JNI/Lucene query-parser failures (e.g. unbalanced quotes in a
            # raw paragraph) shouldn't kill an annotation run — log and
            # return empty so callers can fall back / continue.
            log.warning("Pyserini search failed for query=%r: %s",
                        (query[:120] + "...") if len(query) > 120 else query, e)
            return []

        # Dedup by lowercased paragraph_text (matches upstream OrderedDict trick).
        seen_text: set[str] = set()
        passages: list[Passage] = []
        for hit in hits:
            raw = self._searcher.doc(hit.docid).raw()
            if not raw:
                continue
            try:
                source = json.loads(raw)
            except json.JSONDecodeError:
                # Some prebuilt indices ship docs as bare strings under "contents".
                source = {"contents": raw, "id": hit.docid, "title": "", "paragraph_text": raw}

            title = source.get("title", "") or ""
            text = source.get("paragraph_text") or source.get("contents") or ""
            # DPR-style prebuilt indices (wikipedia-dpr-100w) store
            # ``contents`` as "title\ntext...". Pull out the title heuristically
            # if the source dict didn't ship one.
            if not title and "\n" in text:
                first_line, rest = text.split("\n", 1)
                # Don't split when the first line looks like the body itself.
                if len(first_line) <= 200 and rest.strip():
                    title = first_line.strip()
                    text = rest.strip()
            key = text.strip().lower()
            if key in seen_text:
                continue
            seen_text.add(key)

            passages.append(
                Passage(
                    doc_id=str(source.get("id", hit.docid)),
                    title=title,
                    text=text,
                    score=float(hit.score),
                )
            )
            if len(passages) >= fetch_k:
                break

        if allowed_titles is not None:
            lower_allowed = {t.strip().lower() for t in allowed_titles}
            passages = [p for p in passages if p.title.strip().lower() in lower_allowed]

        return passages[:k]


# ---------------------------------------------------------------------------
# Dataset → retriever resolution
# ---------------------------------------------------------------------------


def _index_path_for_dataset(dataset_name: str, index_root: Path | None = None) -> str:
    """Return the ``index_dir`` argument to pass to ``BM25Retriever``."""
    if dataset_name in _WIKI_DATASETS:
        return f"prebuilt:{PREBUILT_WIKI_INDEX}"
    if dataset_name in _PER_DATASET_INDICES:
        root = index_root if index_root is not None else DEFAULT_INDEX_ROOT
        return str(root / dataset_name)
    raise ValueError(
        f"Unknown dataset_name {dataset_name!r}. "
        f"Expected one of {_WIKI_DATASETS + _PER_DATASET_INDICES}."
    )


@lru_cache(maxsize=8)
def get_retriever(dataset_name: str, index_root: str | None = None) -> BM25Retriever:
    """Return the right BM25 retriever for ``dataset_name``.

    NQ/Trivia/SQuAD share the wiki index. HotpotQA/2Wiki/MuSiQue each have
    their own. Cached per (dataset, index_root) so we don't reload the same
    Lucene index multiple times per process.

    ``index_root`` is the directory containing the per-dataset Lucene index
    folders. Pass ``None`` to use ``DEFAULT_INDEX_ROOT``. Strings (not
    Paths) are accepted so ``functools.lru_cache`` can hash the argument.
    """
    root = Path(index_root) if index_root else None
    return BM25Retriever(_index_path_for_dataset(dataset_name, root))


# ---------------------------------------------------------------------------
# Helper consumed by src/processing/subsample.py
# ---------------------------------------------------------------------------


def find_matching_paragraph_text(
    corpus_name: str,
    original_paragraph_text: str,
    *,
    retriever: BM25Retriever | None = None,
    match_ratio_threshold: int = 95,
) -> dict[str, str] | None:
    """Port of ``processing_scripts/lib.py::find_matching_paragraph_text``.

    Looks up the corpus passage whose text best matches
    ``original_paragraph_text`` using BM25 top-1, then verifies the match
    with ``rapidfuzz.fuzz.partial_ratio`` (threshold 95, same as upstream).

    Returns ``{"title": ..., "paragraph_text": ...}`` or ``None`` if no
    sufficiently-similar passage is found.

    Pass an explicit ``retriever`` to avoid re-resolving the dataset →
    index mapping on every call (useful for batch use).
    """
    try:
        from rapidfuzz import fuzz  # local import — optional dep
    except ImportError as e:
        raise ImportError(
            "rapidfuzz is required for paragraph remapping. "
            "Install with `pip install rapidfuzz`."
        ) from e

    if retriever is None:
        retriever = get_retriever(corpus_name)

    hits = retriever.search(original_paragraph_text, k=1)
    if not hits:
        log.warning("No retrieval result for paragraph (first 80 chars): %r",
                    original_paragraph_text[:80])
        return None

    top = hits[0]
    ratio = fuzz.partial_ratio(original_paragraph_text, top.text)
    if ratio > match_ratio_threshold:
        return {"title": top.title, "paragraph_text": top.text}

    log.debug("Paragraph remap rejected (ratio=%s ≤ %s) for %r",
              ratio, match_ratio_threshold, original_paragraph_text[:80])
    return None


# ---------------------------------------------------------------------------
# Smoke-test entry point (used by scripts/build_retrieval.py step 14)
# ---------------------------------------------------------------------------


def smoke_test_dataset(
    dataset_name: str,
    queries: Iterable[tuple[str, str, str]],
    *,
    k: int = 15,
    index_root: str | None = None,
) -> tuple[int, int]:
    """Run a smoke test against ``dataset_name``'s index.

    ``queries`` is an iterable of ``(qid, query_text, expected_title)``
    tuples. Returns ``(n_pass, n_total)``.
    """
    retriever = get_retriever(dataset_name, index_root=index_root)
    n_pass = 0
    n_total = 0
    for qid, query, expected_title in queries:
        n_total += 1
        passages = retriever.search(query, k=k)
        titles = [p.title for p in passages]
        ok = any(t.strip().lower() == expected_title.strip().lower() for t in titles)
        marker = "OK" if ok else "MISS"
        log.info("  [%s] %-12s qid=%s expected=%r top-%d titles=%s",
                 marker, dataset_name, qid, expected_title, k, titles[:3])
        if ok:
            n_pass += 1
    return n_pass, n_total
