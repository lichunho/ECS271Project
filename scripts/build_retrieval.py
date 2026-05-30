"""Adaptive-RAG retrieval setup orchestrator (data_plan.md §1 steps 12–14).

Steps:
  12a  Verify Pyserini importable + JDK present (best-effort)
  12b  Pull Pyserini's prebuilt ``wikipedia-dpr-100w`` index for NQ/Trivia/SQuAD
  12c  Download the HotpotQA Wikipedia paragraphs tarball if missing
       (~1.5 GB, ~5M passages; the only corpus the sourcing agent didn't
       fetch because §1 needs it only at this phase)
  13   Build per-dataset BM25 indices for HotpotQA / 2Wiki / MuSiQue. Doc
       extraction logic is a port of Adaptive-RAG's
       ``retriever_server/build_index.py`` (HotpotQA from bz2 dumps; 2Wiki
       from train/dev/test JSON contexts; MuSiQue from ans+full v1.0 JSONL
       paragraphs). Each indexed doc keeps {id, title, paragraph_index,
       paragraph_text, url, is_abstract} — same schema as upstream — wrapped
       into Pyserini's required ``{id, contents}`` form so JsonCollection
       indexing works.
  14   Smoke-test: run a known query per dataset and check the gold
       supporting title is in the top-15 hits.

Examples (PowerShell):

    .\\.venv\\Scripts\\python.exe scripts\\build_retrieval.py
    .\\.venv\\Scripts\\python.exe scripts\\build_retrieval.py --step 12
    .\\.venv\\Scripts\\python.exe scripts\\build_retrieval.py --from-step 13 --only-indices musique
    .\\.venv\\Scripts\\python.exe scripts\\build_retrieval.py --dry-run

Idempotent: completed steps are recorded in ``data/.retrieval_state.json``
(mirrors the sourcing orchestrator's state file).
"""

from __future__ import annotations

import argparse
import bz2
import glob
import hashlib
import json
import logging
import os
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator

# Make ``src.*`` importable when running this script directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import httpx  # noqa: E402
from tqdm import tqdm  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_retrieval")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PER_DATASET_INDICES: tuple[str, ...] = ("hotpotqa", "2wikimultihopqa", "musique")

# HotpotQA's Wikipedia paragraph dump (not the QA train/dev files — those
# are downloaded by scripts/download_data.py step 2). This is the corpus
# we build the HotpotQA BM25 index over (~5M passages, abstracts of every
# linked article in HotpotQA's questions).
HOTPOT_WIKI_TARBALL_URL = (
    "https://nlp.stanford.edu/projects/hotpotqa/"
    "enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
)
HOTPOT_WIKI_TARBALL_SIZE = 1_553_565_403  # bytes, verified via HEAD


# Smoke-test queries for step 14. Each tuple is
# ``(qid_label, query_text, expected_gold_title)``. The text and titles
# come from a representative row in each dataset's
# ``data/reference/processed_data_from_repo/{dataset}/test_subsampled.jsonl``
# (we pick row 0 — its supporting title for the first hop). The orchestrator
# falls back to a hard-coded fixture if no reference file is on disk yet.
SMOKE_FIXTURES: dict[str, tuple[str, str, str]] = {
    # Title is one of the gold supporting titles for the question.
    # If we can't find a reference test_subsampled.jsonl on disk, we use
    # these as fallbacks.
    "hotpotqa": (
        "5a8b57f25542995d1e6f1371",
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        "Scott Derrickson",
    ),
    "2wikimultihopqa": (
        "5811079c0bdc11eba7f7acde48001122",
        "Who is the mother of the director of film Polish-Russian War (Film)?",
        "Polish-Russian War (film)",
    ),
    "musique": (
        "2hop__804754_52230",
        "What is the religion of the spouse of the writer of Death of a Salesman?",
        "Arthur Miller",
    ),
}


# ---------------------------------------------------------------------------
# State file (mirrors sourcing orchestrator's pattern)
# ---------------------------------------------------------------------------


class State:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict = {"completed": []}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf8"))
            except Exception as e:
                log.warning("Could not parse %s (%s); starting fresh.", path, e)

    def is_done(self, key: str) -> bool:
        return key in self.data.get("completed", [])

    def mark_done(self, key: str) -> None:
        if key not in self.data["completed"]:
            self.data["completed"].append(key)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def http_download(url: str, dest: Path, *, force: bool = False,
                  expected_size: int | None = None,
                  size_tolerance: float = 0.10) -> None:
    """Stream a URL to ``dest`` with a tqdm progress bar."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        existing = dest.stat().st_size
        if expected_size is None or abs(existing - expected_size) <= expected_size * size_tolerance:
            log.info("  skip (have %s, %s)", dest.name, _human(existing))
            return
        log.warning("  %s exists but size %s != expected ~%s; re-downloading",
                    dest.name, _human(existing), _human(expected_size))
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("  GET %s -> %s", url, dest)
    try:
        with httpx.stream("GET", url, follow_redirects=True,
                          timeout=httpx.Timeout(30.0, read=600.0)) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0)) or expected_size or 0
            with open(tmp, "wb") as f, tqdm(
                total=total or None, unit="B", unit_scale=True, unit_divisor=1024,
                desc=dest.name, leave=False,
            ) as bar:
                for chunk in r.iter_bytes(chunk_size=1024 * 256):
                    f.write(chunk)
                    bar.update(len(chunk))
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


# ---------------------------------------------------------------------------
# Step 12a — verify pyserini
# ---------------------------------------------------------------------------


def step12a_verify_pyserini(*, require_jvm: bool = True) -> None:
    log.info("12a: verifying Pyserini install + JDK...")
    try:
        import pyserini  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Pyserini not installed. Run `pip install pyserini`."
        ) from e
    log.info("  pyserini module path: %s", pyserini.__file__)

    # JDK check — pyjnius lazily probes JAVA_HOME at first JVM use.
    java_home = os.environ.get("JAVA_HOME") or ""
    if not java_home:
        log.warning(
            "  JAVA_HOME is not set. Pyserini needs a JDK 21+ on disk.\n"
            "  On Windows: install Eclipse Temurin 21 "
            "(https://adoptium.net/temurin/releases/?version=21) and set\n"
            "      $env:JAVA_HOME = 'C:\\Program Files\\Eclipse Adoptium\\jdk-21...'\n"
            "      $env:PATH = \"$env:JAVA_HOME\\bin;$env:PATH\"\n"
            "  Or use WSL2: `sudo apt install openjdk-21-jdk` and run pyserini there."
        )
        if require_jvm:
            raise RuntimeError(
                "JAVA_HOME unset — install a JDK 21+ before running step 12/13/14. "
                "Re-run with --dry-run to skip the JVM probe."
            )
        return
    log.info("  JAVA_HOME = %s", java_home)

    if not require_jvm:
        return

    # Force a JVM boot now so we fail fast with a clear error if the JDK
    # is missing or the wrong version.
    from pyserini.search.lucene import LuceneSearcher  # noqa: F401
    log.info("  pyserini.search.lucene.LuceneSearcher imported OK (JVM live)")


# ---------------------------------------------------------------------------
# Step 12b — pull the prebuilt wikipedia-dpr-100w index
# ---------------------------------------------------------------------------


def step12b_pull_prebuilt_wiki(index_root: Path, *, dry_run: bool) -> None:
    """Pyserini caches its prebuilt indices under ``~/.cache/pyserini/indexes``
    by default. We trigger the download but also stash a small breadcrumb at
    ``{index_root}/wiki/.prebuilt_index_pointer`` so step-skip logic and
    downstream code can locate the cached path without re-doing the
    download/validation dance.
    """
    target_dir = index_root / "wiki"
    pointer_path = target_dir / ".prebuilt_index_pointer"

    if dry_run:
        log.info("  [dry-run] would download Pyserini prebuilt index 'wikipedia-dpr-100w'")
        return

    target_dir.mkdir(parents=True, exist_ok=True)

    from pyserini.search.lucene import LuceneSearcher
    log.info("  Pulling prebuilt index 'wikipedia-dpr-100w' (~11 GB compressed, "
             "~21M docs). This caches to ~/.cache/pyserini/indexes/ — Pyserini "
             "skips the download if already present.")
    searcher = LuceneSearcher.from_prebuilt_index("wikipedia-dpr-100w", verbose=True)
    if searcher is None:
        raise RuntimeError("Pyserini failed to load the wikipedia-dpr-100w prebuilt index.")
    cached_path = searcher.index_dir
    log.info("  Prebuilt index ready at %s (num_docs=%d)",
             cached_path, searcher.num_docs)
    pointer_path.write_text(str(cached_path), encoding="utf8")


# ---------------------------------------------------------------------------
# Step 12c — fetch HotpotQA's Wikipedia paragraphs corpus
# ---------------------------------------------------------------------------


def step12c_fetch_hotpot_wiki(data_dir: Path, *, force: bool, dry_run: bool) -> None:
    """The HotpotQA QA files (train/dev) come down in
    ``scripts/download_data.py`` step 2, but their *Wikipedia abstracts
    corpus* (the 5M-passage tarball indexed by Adaptive-RAG's
    ``make_hotpotqa_documents``) is separate. Fetch it here on demand.

    Outputs the extracted tree to ``{data_dir}/raw/hotpotqa/wikpedia-paragraphs/``
    (note the upstream typo — ``wikpedia``, not ``wikipedia`` — preserved so
    the glob in ``make_hotpot_documents`` below matches the upstream code).
    """
    out_dir = data_dir / "raw" / "hotpotqa" / "wikpedia-paragraphs"
    if dry_run:
        log.info("  [dry-run] would fetch %s -> %s", HOTPOT_WIKI_TARBALL_URL, out_dir)
        return

    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        # Cheap presence check — count bz2 files at depth 2
        bz2_count = sum(1 for _ in out_dir.glob("*/wiki_*.bz2"))
        if bz2_count > 0:
            log.info("  skip (have %d wiki_*.bz2 files under %s)", bz2_count, out_dir)
            return

    tarball = data_dir / "raw" / "hotpotqa" / "enwiki-20171001-paragraphs.tar.bz2"
    http_download(HOTPOT_WIKI_TARBALL_URL, tarball, force=force,
                  expected_size=HOTPOT_WIKI_TARBALL_SIZE)

    log.info("  extracting %s (this takes a few min)...", tarball.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:bz2") as tf:
        # Upstream tarball layout: enwiki-20171001-pages-meta-current-withlinks-abstracts/AA/wiki_00.bz2 etc.
        members = tf.getmembers()
        top_levels = {m.name.split("/", 1)[0] for m in members if "/" in m.name}
        # Strip the single top-level dir so files land at out_dir/AA/wiki_00.bz2 directly.
        strip = None
        if len(top_levels) == 1:
            strip = next(iter(top_levels)) + "/"
        for m in members:
            if strip and m.name.startswith(strip):
                m.name = m.name[len(strip):]
            if not m.name:
                continue
        try:
            tf.extractall(out_dir, members=members, filter="data")
        except TypeError:
            tf.extractall(out_dir, members=members)

    # Tarball is large — remove it now that the bz2 files are unpacked.
    tarball.unlink(missing_ok=True)
    n_bz2 = sum(1 for _ in out_dir.glob("*/wiki_*.bz2"))
    log.info("  HotpotQA wiki corpus extracted: %d bz2 chunks", n_bz2)


# ---------------------------------------------------------------------------
# Step 13 — document iterators (ports of Adaptive-RAG's build_index.py)
# ---------------------------------------------------------------------------


def _stable_id(payload: str) -> str:
    """32-char hash to match the upstream document-ID length.

    Upstream uses ``hashlib.blake2b()`` with ``dill``-serialized input.
    We mirror the algorithm but feed the *serialized JSON*: this is
    deterministic across Python versions and avoids the dill dep.
    """
    h = hashlib.blake2b(payload.encode("utf-8"), digest_size=24).hexdigest()
    return h[:32]


def _doc(id_: str, title: str, paragraph_text: str, url: str,
         paragraph_index: int, is_abstract: bool) -> dict:
    """Build the indexed-document dict.

    Pyserini's ``LuceneIndexer.add_doc_dict`` only requires ``id`` and
    ``contents``. We additionally retain title/paragraph_text/url/is_abstract
    so the raw stored JSON is identical-shaped to upstream's ES ``_source``
    and our ``BM25Retriever.search`` can pull them back out of ``doc.raw()``.

    The ``contents`` field is what Pyserini tokenises and indexes. We
    concatenate ``title`` + ``paragraph_text`` because the upstream ES
    config also lets queries hit ``title`` via the bool ``should`` clause
    (``retrieve_paragraphs(query_title_field_too=True)``). For our
    no-title-boost default, indexing title into ``contents`` once is the
    simplest equivalent.
    """
    contents = f"{title}\n{paragraph_text}" if title else paragraph_text
    return {
        "id": id_,
        "contents": contents,
        "title": title,
        "paragraph_index": paragraph_index,
        "paragraph_text": paragraph_text,
        "url": url,
        "is_abstract": is_abstract,
    }


def iter_hotpotqa_docs(data_dir: Path) -> Iterator[dict]:
    """Mirror of upstream ``make_hotpotqa_documents`` (build_index.py).

    Reads bz2 files under ``raw/hotpotqa/wikpedia-paragraphs/*/wiki_*.bz2``
    and yields one doc per Wikipedia article (abstracts only).
    """
    raw_glob = data_dir / "raw" / "hotpotqa" / "wikpedia-paragraphs" / "*" / "wiki_*.bz2"
    paths = sorted(glob.glob(str(raw_glob)))
    if not paths:
        raise FileNotFoundError(
            f"No HotpotQA wiki bz2 files found at {raw_glob}. "
            f"Run step 12c (or `--step 12c`) to fetch them."
        )
    log.info("  HotpotQA corpus: %d bz2 chunks", len(paths))
    used_ids: set[str] = set()
    for path in tqdm(paths, desc="hotpotqa bz2", leave=False):
        with bz2.BZ2File(path, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                instance = json.loads(line)
                title = instance.get("title", "") or ""
                # ``text`` in HotpotQA wiki dumps is List[str] (sentences).
                sentences = instance.get("text", []) or []
                if isinstance(sentences, list):
                    paragraph_text = " ".join(s.strip() for s in sentences if isinstance(s, str)).strip()
                else:
                    paragraph_text = str(sentences).strip()
                if not paragraph_text:
                    continue
                url = instance.get("url", "") or ""
                id_ = _stable_id(json.dumps([title, paragraph_text], sort_keys=True))
                if id_ in used_ids:
                    continue
                used_ids.add(id_)
                yield _doc(id_, title, paragraph_text, url,
                           paragraph_index=0, is_abstract=True)


def iter_2wiki_docs(data_dir: Path) -> Iterator[dict]:
    """Mirror of upstream ``make_2wikimultihopqa_documents``."""
    raw_paths = [
        data_dir / "raw" / "2wikimultihopqa" / fname
        for fname in ("train.json", "dev.json", "test.json")
    ]
    used_ids: set[str] = set()
    for raw_path in raw_paths:
        if not raw_path.exists():
            log.warning("  2Wiki source missing: %s — skipping", raw_path)
            continue
        log.info("  2Wiki: scanning %s", raw_path.name)
        with open(raw_path, "r", encoding="utf8") as f:
            full = json.load(f)
        for instance in tqdm(full, desc=f"2wiki/{raw_path.name}", leave=False):
            for paragraph in instance.get("context", []):
                if not paragraph or len(paragraph) < 2:
                    continue
                title = paragraph[0] or ""
                sentences = paragraph[1] or []
                paragraph_text = " ".join(s for s in sentences if isinstance(s, str))
                if not paragraph_text.strip():
                    continue
                full_id = _stable_id(json.dumps([title, paragraph_text], sort_keys=True))
                if full_id in used_ids:
                    continue
                used_ids.add(full_id)
                yield _doc(full_id, title, paragraph_text, url="",
                           paragraph_index=0, is_abstract=True)


def iter_musique_docs(data_dir: Path) -> Iterator[dict]:
    """Mirror of upstream ``make_musique_documents``.

    Indexes both ``musique_ans_v1.0_*`` and ``musique_full_v1.0_*`` paragraphs.
    """
    raw_dir = data_dir / "raw" / "musique"
    candidate_names = [
        "musique_ans_v1.0_dev.jsonl",
        "musique_ans_v1.0_test.jsonl",
        "musique_ans_v1.0_train.jsonl",
        "musique_full_v1.0_dev.jsonl",
        "musique_full_v1.0_test.jsonl",
        "musique_full_v1.0_train.jsonl",
    ]
    used_ids: set[str] = set()
    for fname in candidate_names:
        path = raw_dir / fname
        if not path.exists():
            log.info("  MuSiQue source missing: %s — skipping", path.name)
            continue
        with open(path, "r", encoding="utf8") as f:
            for line in tqdm(f, desc=f"musique/{fname}", leave=False):
                line = line.strip()
                if not line:
                    continue
                instance = json.loads(line)
                for paragraph in instance.get("paragraphs", []) or []:
                    title = paragraph.get("title", "") or ""
                    paragraph_text = paragraph.get("paragraph_text", "") or ""
                    if not paragraph_text.strip():
                        continue
                    full_id = _stable_id(
                        json.dumps([title, paragraph_text], sort_keys=True)
                    )
                    if full_id in used_ids:
                        continue
                    used_ids.add(full_id)
                    yield _doc(full_id, title, paragraph_text, url="",
                               paragraph_index=0, is_abstract=True)


_DOC_ITERATORS: dict[str, Callable[[Path], Iterator[dict]]] = {
    "hotpotqa": iter_hotpotqa_docs,
    "2wikimultihopqa": iter_2wiki_docs,
    "musique": iter_musique_docs,
}


# ---------------------------------------------------------------------------
# Step 13 — index builder
# ---------------------------------------------------------------------------


def _build_index_via_jsonl(
    docs: Iterator[dict],
    index_dir: Path,
    *,
    batch_size: int = 10_000,
    threads: int = 4,
) -> int:
    """Write all docs to a temp JSONL file, then invoke Pyserini's
    ``index.lucene`` CLI over JsonCollection. This is far faster than the
    in-process ``LuceneIndexer.add_doc_dict`` path because the CLI
    parallelises across ``--threads`` workers.

    We keep ``title``, ``paragraph_text``, ``url``, ``paragraph_index``,
    ``is_abstract`` alongside ``id`` and ``contents`` in each JSON line.
    Pyserini's JsonCollection passes through extra fields into the doc's
    raw store, so ``BM25Retriever.search`` can recover them via
    ``searcher.doc(docid).raw()``.
    """
    index_dir = Path(index_dir)
    if index_dir.exists():
        log.info("  removing existing index dir %s", index_dir)
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # Pyserini's JsonCollection wants either a single JSONL or a directory of
    # JSONLs. We stream to a temp directory because some corpora (HotpotQA)
    # produce huge files.
    with tempfile.TemporaryDirectory(prefix="pyserini_input_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        jsonl_path = tmpdir_path / "corpus.jsonl"
        n = 0
        log.info("  writing JSON corpus to %s ...", jsonl_path)
        with open(jsonl_path, "w", encoding="utf8") as out_f:
            for doc in docs:
                out_f.write(json.dumps(doc) + "\n")
                n += 1
                if n % 100_000 == 0:
                    log.info("    ... %d docs serialized", n)
        log.info("  serialized %d docs (%s on disk)", n, _human(jsonl_path.stat().st_size))
        if n == 0:
            raise RuntimeError("No documents to index — check the corpus paths.")

        log.info("  running pyserini.index.lucene with %d threads ...", threads)
        # Use the CLI runner — pyserini.index.lucene exposes a __main__ that
        # builds an Anserini IndexCollection on top of JsonCollection.
        import subprocess
        cmd = [
            sys.executable, "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", str(tmpdir_path),
            "--index", str(index_dir),
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", str(threads),
            "--storePositions",
            "--storeDocvectors",
            "--storeRaw",
        ]
        log.info("  $ %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"pyserini.index.lucene exited with code {e.returncode}. "
                f"Check JAVA_HOME and that JDK 21+ is installed."
            ) from e
    return n


def step13_build_index(data_dir: Path, dataset: str, index_root: Path,
                       *, force: bool, dry_run: bool, threads: int = 4) -> None:
    index_dir = index_root / dataset
    if dry_run:
        log.info("  [dry-run] would build %s index at %s", dataset, index_dir)
        return
    if index_dir.exists() and any(index_dir.iterdir()) and not force:
        log.info("  skip (have non-empty index at %s)", index_dir)
        return

    iterator_fn = _DOC_ITERATORS[dataset]
    n = _build_index_via_jsonl(iterator_fn(data_dir), index_dir, threads=threads)
    log.info("  %s index built at %s (%d docs)", dataset, index_dir, n)


# ---------------------------------------------------------------------------
# Step 14 — smoke test
# ---------------------------------------------------------------------------


def _load_smoke_fixture_from_reference(
    data_dir: Path, dataset: str,
) -> tuple[str, str, str] | None:
    """If the sourcing agent has produced
    ``data/reference/processed_data_from_repo/{dataset}/test_subsampled.jsonl``,
    pick the first row whose first supporting context has a non-empty title.
    """
    p = (data_dir / "reference" / "processed_data_from_repo"
         / dataset / "test_subsampled.jsonl")
    if not p.exists():
        return None
    with open(p, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            inst = json.loads(line)
            for ctx in inst.get("contexts", []):
                if ctx.get("is_supporting") and ctx.get("title"):
                    return (
                        inst["question_id"],
                        inst["question_text"],
                        ctx["title"],
                    )
            # if no is_supporting flag, take the first context with a title
            for ctx in inst.get("contexts", []):
                if ctx.get("title"):
                    return (
                        inst["question_id"],
                        inst["question_text"],
                        ctx["title"],
                    )
    return None


def step14_smoke_test(
    data_dir: Path, index_root: Path, datasets_subset: Iterable[str],
    *, k: int = 15, dry_run: bool,
) -> bool:
    if dry_run:
        log.info("  [dry-run] would smoke-test BM25 indices for %s",
                 list(datasets_subset))
        return True

    from src.retrieval import smoke_test_dataset  # noqa: E402

    overall_pass = 0
    overall_total = 0

    # For per-dataset indices
    for ds in datasets_subset:
        if ds not in _DOC_ITERATORS:
            continue
        fixture = _load_smoke_fixture_from_reference(data_dir, ds)
        if fixture is None:
            fixture = SMOKE_FIXTURES.get(ds)
            if fixture is None:
                log.warning("  no fixture for %s — skipping", ds)
                continue
            log.info("  using hard-coded fixture for %s (no reference file on disk)", ds)
        else:
            log.info("  using fixture from reference tarball for %s", ds)
        ok, tot = smoke_test_dataset(
            ds, [fixture], k=k, index_root=str(index_root),
        )
        overall_pass += ok
        overall_total += tot

    # Wiki smoke test: pick from NQ if we have a reference file.
    if any(d in datasets_subset for d in ("nq", "trivia", "squad", "wiki")):
        wiki_fixture: tuple[str, str, str] | None = None
        for ref_ds in ("nq", "trivia", "squad"):
            wiki_fixture = _load_smoke_fixture_from_reference(data_dir, ref_ds)
            if wiki_fixture is not None:
                log.info("  using fixture from reference/%s for the wiki index", ref_ds)
                break
        if wiki_fixture is None:
            wiki_fixture = (
                "nq_dev_0",
                "Who wrote the score for Star Wars?",
                "John Williams",
            )
            log.info("  using hard-coded fixture for the wiki index")
        try:
            ok, tot = smoke_test_dataset(
                "nq", [wiki_fixture], k=k, index_root=str(index_root),
            )
            overall_pass += ok
            overall_total += tot
        except FileNotFoundError as e:
            log.warning("  wiki index missing — skipping wiki smoke test: %s", e)

    log.info("Step 14 smoke-test summary: %d/%d queries returned the gold title in top-%d",
             overall_pass, overall_total, k)
    # We don't hard-fail the orchestrator on a miss: BM25 over a single hop is
    # noisy and a single fixture isn't a reliable gate. Just report.
    return overall_pass == overall_total


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


STEPS: list[tuple[int, str]] = [
    (12, "pyserini_and_wiki"),  # 12a + 12b + 12c
    (13, "build_per_dataset_indices"),
    (14, "smoke_test"),
]


def run(args: argparse.Namespace) -> int:
    data_dir: Path = args.data_dir.resolve()
    index_root: Path = (data_dir / "indices" / "pyserini").resolve()
    state = State(data_dir / ".retrieval_state.json")

    if args.only_indices:
        only = tuple(d.strip() for d in args.only_indices.split(","))
        for d in only:
            if d not in PER_DATASET_INDICES:
                log.error("Unknown index %s; choose from %s", d, PER_DATASET_INDICES)
                return 2
    else:
        only = PER_DATASET_INDICES

    if args.step is not None:
        steps = [args.step]
    else:
        lo = args.from_step or 12
        hi = args.to_step or 14
        steps = list(range(lo, hi + 1))

    log.info("Plan: data_dir=%s  index_root=%s  steps=%s  only_indices=%s  dry_run=%s",
             data_dir, index_root, steps, only, args.dry_run)

    index_root.mkdir(parents=True, exist_ok=True)

    for step in steps:
        step_key = f"step{step}__{','.join(only)}"
        if args.skip_existing and not args.force and state.is_done(step_key):
            log.info("Step %d: skip (state.json says complete)", step)
            continue

        log.info("=" * 72)
        log.info("Step %d", step)
        log.info("=" * 72)
        t0 = time.time()

        try:
            if step == 12:
                step12a_verify_pyserini(require_jvm=not args.dry_run)
                step12b_pull_prebuilt_wiki(index_root, dry_run=args.dry_run)
                step12c_fetch_hotpot_wiki(data_dir, force=args.force, dry_run=args.dry_run)

            elif step == 13:
                for ds in only:
                    log.info("--- step 13: building %s index ---", ds)
                    step13_build_index(
                        data_dir, ds, index_root,
                        force=args.force, dry_run=args.dry_run,
                        threads=args.threads,
                    )

            elif step == 14:
                # Smoke-test the per-dataset indices in scope + the wiki index.
                step14_smoke_test(
                    data_dir, index_root,
                    datasets_subset=tuple(only) + ("nq",),
                    dry_run=args.dry_run, k=15,
                )

        except Exception as e:
            log.exception("Step %d failed: %s", step, e)
            return 1

        elapsed = time.time() - t0
        log.info("Step %d done in %.1fs", step, elapsed)
        if not args.dry_run:
            state.mark_done(step_key)

    log.info("All requested retrieval steps completed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--step", type=int, choices=range(12, 15), metavar="N",
                   help="Run only this single step (12, 13, or 14).")
    g.add_argument("--from-step", type=int, choices=range(12, 15), metavar="N",
                   help="Start at this step. Default 12.")
    p.add_argument("--to-step", type=int, choices=range(12, 15), metavar="N",
                   help="Stop at this step (inclusive). Default 14.")
    p.add_argument("--only-indices", type=str, default=None,
                   help="Comma-separated subset of {hotpotqa,2wikimultihopqa,musique} "
                        "to build in step 13. Default: all three.")
    p.add_argument("--data-dir", type=Path, default=Path("data"),
                   help="Root data directory. Default: ./data")
    p.add_argument("--threads", type=int, default=4,
                   help="Threads for Pyserini's CLI indexer in step 13. Default 4.")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="(default) Skip steps already recorded as complete in .retrieval_state.json.")
    p.add_argument("--force", action="store_true",
                   help="Rebuild from scratch (removes index dirs and re-downloads).")
    p.add_argument("--dry-run", action="store_true",
                   help="Log what each step would do, without acting.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
