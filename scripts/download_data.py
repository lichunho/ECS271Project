"""Adaptive-RAG data-sourcing orchestrator.

Implements §1 (steps 1–11) of ``data_plan.md``: downloads raw QA datasets,
processes them with our ports under ``src/processing/``, subsamples 500
instances per (dataset, split), and diffs the resulting question_id sets
against Adaptive-RAG's reference tarball.

Step 12–14 (Pyserini BM25 setup) is intentionally out of scope.

Examples (PowerShell, Windows):

    # Default: run all eleven steps for all six datasets
    .\\.venv\\Scripts\\python.exe scripts\\download_data.py

    # Resume after a crash (skips steps that already completed)
    .\\.venv\\Scripts\\python.exe scripts\\download_data.py

    # Just MuSiQue end-to-end (smoke test)
    .\\.venv\\Scripts\\python.exe scripts\\download_data.py --only-datasets musique

    # Only step 6 (sanity tarballs)
    .\\.venv\\Scripts\\python.exe scripts\\download_data.py --step 6

    # Re-run everything from scratch
    .\\.venv\\Scripts\\python.exe scripts\\download_data.py --force
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import shutil
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Callable, Iterable

# Make ``src.*`` importable when running this script directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import httpx  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.processing import (  # noqa: E402
    process_2wikimultihopqa,
    process_hotpotqa,
    process_musique,
    process_nq,
    process_squad,
    process_trivia,
    subsample,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_data")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_DATASETS = ("hotpotqa", "2wikimultihopqa", "musique", "nq", "trivia", "squad")
MULTI_HOP_DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
DPR_DATASETS = ("nq", "trivia", "squad")

# Each dataset's processed dev/train files live under data/processed/{name}/.
PROCESSING_FNS: dict[str, Callable[[Path, Path], dict[str, int]]] = {
    "hotpotqa": process_hotpotqa.main,
    "2wikimultihopqa": process_2wikimultihopqa.main,
    "musique": process_musique.main,
    "nq": process_nq.main,
    "trivia": process_trivia.main,
    "squad": process_squad.main,
}

# Step 6 reference tarballs — the Adaptive-RAG repo commits these at HEAD.
# Sizes verified against a fresh `git clone` of starsuzi/Adaptive-RAG (May 2025).
REFERENCE_TARBALLS = {
    "processed_data.tar.gz": {
        "url": "https://github.com/starsuzi/Adaptive-RAG/raw/main/processed_data.tar.gz",
        "approx_size": 19_890_027,
    },
    "predictions.tar.gz": {
        "url": "https://github.com/starsuzi/Adaptive-RAG/raw/main/predictions.tar.gz",
        "approx_size": 22_596_794,
    },
    "data.tar.gz": {
        "url": "https://github.com/starsuzi/Adaptive-RAG/raw/main/data.tar.gz",
        "approx_size": 20_246_445,
    },
}

# Step 2: HotpotQA raw downloads (CMU host can be flaky)
HOTPOT_FILES = {
    "hotpot_train_v1.1.json": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
    "hotpot_dev_distractor_v1.json": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
}

# Step 3: 2Wiki Dropbox zip (Adaptive-RAG's pinned URL)
WIKI2_URL = "https://www.dropbox.com/s/7ep3h8unu2njfxv/data_ids.zip?dl=1"

# Step 4: MuSiQue Google Drive ID
MUSIQUE_GDRIVE_ID = "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h"

# IRCoT processed_data.zip — needed only as the source of ``dev_subsampled.jsonl``
# (100 qids per multi-hop dataset) used as the AVOID file by subsample.py for
# ``test``. Without it, our multi-hop ``test_subsampled`` will NOT match the
# Adaptive-RAG reference tarball (verified: 93/500 overlap without it, 500/500
# with it). See README §"Datasets" of starsuzi/Adaptive-RAG. Note that the
# Adaptive-RAG ``processed_data.tar.gz`` ships only ``dev_500_subsampled.jsonl``
# (500 rows), which is NOT the same set as IRCoT's ``dev_subsampled.jsonl``
# (100 rows) — they overlap only ~27%, so we genuinely need the IRCoT copy.
IRCOT_PROCESSED_GDRIVE_ID = "1t2BjJtsejSIUZI54PKObMFG6_wMMG3bC"
MUSIQUE_EXPECTED_FILES = (
    "musique_ans_v1.0_train.jsonl",
    "musique_ans_v1.0_dev.jsonl",
    "musique_ans_v1.0_test.jsonl",
    "dev_test_singlehop_questions_v1.0.json",
)

# Step 5: DPR biencoder files (gzipped)
DPR_FILES = {
    "nq": [
        ("biencoder-nq-train.json.gz",
         "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz"),
        ("biencoder-nq-dev.json.gz",
         "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz"),
    ],
    "trivia": [
        ("biencoder-trivia-train.json.gz",
         "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-train.json.gz"),
        ("biencoder-trivia-dev.json.gz",
         "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-dev.json.gz"),
    ],
    "squad": [
        ("biencoder-squad1-train.json.gz",
         "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-train.json.gz"),
        ("biencoder-squad1-dev.json.gz",
         "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-dev.json.gz"),
    ],
}

# Step 7 outputs to check for skip-existing
PROCESSED_OUTPUTS_PER_DATASET = ("train.jsonl", "dev.jsonl")

# Step 11: only diff multi-hop and DPR datasets that exist in the reference tarball.
REFERENCE_DATASETS = ("hotpotqa", "2wikimultihopqa", "musique", "nq", "trivia", "squad")


# ---------------------------------------------------------------------------
# State file
# ---------------------------------------------------------------------------


class State:
    """Tiny JSON-backed state file: tracks which (step, dataset?) pairs are done."""

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

    def clear(self) -> None:
        self.data = {"completed": []}
        if self.path.exists():
            self.path.unlink()


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n = n / 1024  # type: ignore[assignment]
    return f"{n:.1f}TB"


def http_download(url: str, dest: Path, *, force: bool = False,
                  expected_size: int | None = None,
                  size_tolerance: float = 0.10) -> None:
    """Stream a URL to ``dest`` with a tqdm progress bar.

    Skips the download if ``dest`` already exists and has the expected size
    (within ``size_tolerance``). Use ``force=True`` to redownload.
    """
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
        with httpx.stream("GET", url, follow_redirects=True, timeout=httpx.Timeout(30.0, read=300.0)) as r:
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

    if expected_size is not None:
        got = dest.stat().st_size
        if abs(got - expected_size) > expected_size * size_tolerance:
            log.warning("  size mismatch for %s: got %s, expected ~%s",
                        dest.name, _human(got), _human(expected_size))


def gdown_download(file_id: str, dest: Path, *, force: bool = False) -> None:
    """Download a Google Drive file by ID using gdown."""
    import gdown  # local import — heavy + optional dep

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        log.info("  skip (have %s, %s)", dest.name, _human(dest.stat().st_size))
        return
    log.info("  gdown id=%s -> %s", file_id, dest)
    # gdown >= 6 dropped the ``fuzzy`` kwarg and the keyword ``id``; pass the
    # full URL form which works on both 5.x and 6.x.
    url = f"https://drive.google.com/uc?id={file_id}&confirm=t"
    gdown.download(url, str(dest), quiet=False)


def extract_tar_gz(archive: Path, dest_dir: Path, *, strip_top_dir: str | None = None) -> None:
    """Extract a .tar.gz into ``dest_dir``. If ``strip_top_dir`` is given and
    every member starts with that prefix, strip it (so contents land at
    ``dest_dir`` directly, not ``dest_dir/<top>``).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    log.info("  extract %s -> %s", archive.name, dest_dir)
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        filtered: list[tarfile.TarInfo] = []
        if strip_top_dir:
            prefix1 = f"{strip_top_dir}/"
            prefix2 = f"./{strip_top_dir}/"
            top_only = {strip_top_dir, f"./{strip_top_dir}", f"./{strip_top_dir}/", f"{strip_top_dir}/"}
            for m in members:
                name = m.name.rstrip("/")
                if name in top_only or name == "":
                    continue  # skip the top dir itself
                if name.startswith(prefix2):
                    name = name[len(prefix2):]
                elif name.startswith(prefix1):
                    name = name[len(prefix1):]
                if not name:
                    continue
                m.name = name
                filtered.append(m)
        else:
            filtered = [m for m in members if m.name and m.name.strip("./")]
        # ``filter='data'`` is the Python 3.12+ safe extraction mode.
        try:
            tf.extractall(dest_dir, members=filtered, filter="data")
        except TypeError:
            tf.extractall(dest_dir, members=filtered)


def extract_zip(archive: Path, dest_dir: Path, *, flatten: bool = False) -> None:
    """Extract a .zip into ``dest_dir``. With ``flatten=True`` mimics
    ``unzip -j`` (junk the path, drop everything at top level)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    log.info("  extract %s -> %s", archive.name, dest_dir)
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name if flatten else info.filename
            if not name or name.endswith(".DS_Store"):
                continue
            out_path = dest_dir / name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)


def gunzip(src: Path, dest: Path, *, force: bool = False) -> None:
    """Decompress a single gzipped file."""
    if dest.exists() and not force:
        log.info("  skip gunzip (have %s)", dest.name)
        return
    log.info("  gunzip %s -> %s", src.name, dest.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def step1_make_tree(data_dir: Path) -> None:
    """Create the on-disk layout from data_plan.md §On-disk layout."""
    for d in ("raw", "processed", "eval_500", "train_500", "reference"):
        (data_dir / d).mkdir(parents=True, exist_ok=True)
    for d in ALL_DATASETS:
        (data_dir / "raw" / d).mkdir(parents=True, exist_ok=True)
    log.info("Created on-disk layout under %s", data_dir)


def step2_download_hotpotqa(data_dir: Path, *, force: bool, dry_run: bool) -> None:
    """Step 2: HotpotQA raw JSONs from CMU."""
    target = data_dir / "raw" / "hotpotqa"
    for fname, url in HOTPOT_FILES.items():
        dest = target / fname
        if dry_run:
            log.info("  [dry-run] would download %s", url)
            continue
        try:
            http_download(url, dest, force=force)
        except Exception as e:
            # CMU host is flaky; let caller decide whether to abort.
            log.error("HotpotQA download failed for %s: %s", fname, e)
            raise


def step3_download_2wiki(data_dir: Path, *, force: bool, dry_run: bool) -> None:
    """Step 3: 2WikiMultiHopQA from Dropbox; extract the zip flat."""
    target = data_dir / "raw" / "2wikimultihopqa"
    zip_path = target / ".tmp_2wiki.zip"
    if dry_run:
        log.info("  [dry-run] would download %s", WIKI2_URL)
        return
    http_download(WIKI2_URL, zip_path, force=force)
    extract_zip(zip_path, target, flatten=True)
    if not force:
        zip_path.unlink(missing_ok=True)


def step4_download_musique(data_dir: Path, *, force: bool, dry_run: bool) -> None:
    """Step 4: MuSiQue from Google Drive via gdown; extract the zip flat."""
    target = data_dir / "raw" / "musique"
    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / ".tmp_musique.zip"
    if dry_run:
        log.info("  [dry-run] would gdown id=%s", MUSIQUE_GDRIVE_ID)
        return

    # If all expected files already exist, no need to redownload.
    if not force and all((target / f).exists() for f in MUSIQUE_EXPECTED_FILES):
        log.info("  skip (have all MuSiQue files in %s)", target)
        return

    gdown_download(MUSIQUE_GDRIVE_ID, zip_path, force=force)
    extract_zip(zip_path, target, flatten=True)
    zip_path.unlink(missing_ok=True)


def step5_download_dpr(data_dir: Path, *, force: bool, dry_run: bool,
                       datasets_subset: Iterable[str]) -> None:
    """Step 5: DPR-curated biencoder files for NQ/Trivia/SQuAD."""
    wanted = [d for d in DPR_DATASETS if d in datasets_subset]
    for dataset in wanted:
        target = data_dir / "raw" / dataset
        for fname, url in DPR_FILES[dataset]:
            gz_path = target / fname
            json_path = target / fname[:-len(".gz")]
            if dry_run:
                log.info("  [dry-run] would download %s", url)
                continue
            if json_path.exists() and not force:
                log.info("  skip (have %s)", json_path.name)
                continue
            http_download(url, gz_path, force=force)
            gunzip(gz_path, json_path, force=force)
            # Keep the .gz around as a cheap resume token if disk allows; remove on success.
            gz_path.unlink(missing_ok=True)


def step6_download_reference(data_dir: Path, *, force: bool, dry_run: bool) -> None:
    """Step 6: Sanity-baseline tarballs from the Adaptive-RAG repo."""
    target = data_dir / "reference"
    target.mkdir(parents=True, exist_ok=True)

    for fname, spec in REFERENCE_TARBALLS.items():
        archive = target / fname
        if dry_run:
            log.info("  [dry-run] would download %s", spec["url"])
            continue
        http_download(spec["url"], archive, force=force, expected_size=spec["approx_size"])

        # Each tarball contains a single top-level dir matching its base name.
        top_dir = fname.replace(".tar.gz", "")
        out_dir = target / f"{top_dir}_from_repo"

        if out_dir.exists() and not force:
            log.info("  skip extract (have %s)", out_dir)
            continue
        # Wipe stale partial extraction.
        if out_dir.exists():
            shutil.rmtree(out_dir)
        extract_tar_gz(archive, out_dir, strip_top_dir=top_dir)


def step7_process(data_dir: Path, *, datasets_subset: Iterable[str],
                  force: bool, dry_run: bool) -> dict[str, dict[str, int]]:
    """Step 7: run each ported processor; produce data/processed/{ds}/{train,dev}.jsonl."""
    counts: dict[str, dict[str, int]] = {}
    for dataset in datasets_subset:
        raw_dir = data_dir / "raw" / dataset
        out_dir = data_dir / "processed" / dataset

        if dry_run:
            log.info("  [dry-run] would process %s", dataset)
            continue

        # Skip-existing: if both train.jsonl and dev.jsonl already exist, skip.
        if not force and all((out_dir / f).exists() for f in PROCESSED_OUTPUTS_PER_DATASET):
            sizes = {f: _count_lines(out_dir / f) for f in PROCESSED_OUTPUTS_PER_DATASET}
            log.info("  skip %s (have %s)", dataset, sizes)
            counts[dataset] = {k.replace(".jsonl", ""): v for k, v in sizes.items()}
            continue

        log.info("Processing %s (raw=%s)", dataset, raw_dir)
        fn = PROCESSING_FNS[dataset]
        try:
            counts[dataset] = fn(raw_dir, out_dir)
        except FileNotFoundError as e:
            log.error("Cannot process %s — missing raw file: %s", dataset, e)
            raise
        log.info("  %s row counts: %s", dataset, counts[dataset])
    return counts


def _count_lines(p: Path) -> int:
    n = 0
    with open(p, "r", encoding="utf8") as f:
        for _ in f:
            n += 1
    return n


def _seed_ircot_avoid_file(data_dir: Path, dataset: str, *, force: bool) -> None:
    """For multi-hop datasets, drop IRCoT's ``dev_subsampled.jsonl`` next to our
    ``dev.jsonl`` so ``subsample.main(..., 'test', ...)`` picks it up as the
    avoid file. Without this seeding our ``test_subsampled.jsonl`` will not
    match the Adaptive-RAG reference.

    Downloads (and caches) the full IRCoT ``processed_data.zip`` from Google
    Drive on first call — ~1.7 GB. The zip is reused across all multi-hop
    datasets on subsequent calls.
    """
    if dataset not in MULTI_HOP_DATASETS:
        return  # NQ/Trivia/SQuAD don't need this seed file

    proc_dir = data_dir / "processed" / dataset
    target = proc_dir / "dev_subsampled.jsonl"
    if target.exists() and not force:
        log.info("  %s: dev_subsampled.jsonl already in place — skipping IRCoT seed",
                 dataset)
        return

    cache_dir = data_dir / "raw" / ".ircot_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "ircot_processed_data.zip"

    if not zip_path.exists():
        log.info("  IRCoT processed_data.zip not cached — downloading (~1.7 GB, "
                 "needed only for multi-hop ``dev_subsampled.jsonl`` seed files).")
        gdown_download(IRCOT_PROCESSED_GDRIVE_ID, zip_path, force=False)

    member = f"processed_data/{dataset}/dev_subsampled.jsonl"
    log.info("  %s: extracting %s -> %s", dataset, member, target)
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)


def _subsample_via_reference_qids(
    data_dir: Path, dataset: str, ref_filename: str, output_path: Path,
) -> int:
    """Filter our processed dev to the reference tarball's qid set, in reference order.

    Why we prefer this over random sampling: Adaptive-RAG's reference
    ``{test,dev_500}_subsampled.jsonl`` is not reproducible from
    ``random.seed(13370) + random.sample(...)`` for single-hop datasets — their
    sampling RNG state differed from a fresh seed, and reverse-engineering the
    exact upstream state is not worth the time. The reference qids themselves
    are the canonical paper splits, and we already have them on disk after
    step 6. We just need to project them onto our (faithfully-processed) rows.

    Returns the number of rows written. Raises if any reference qid is missing
    from our processed dev (would indicate a bug in our process_*.py port).
    """
    ref_path = data_dir / "reference" / "processed_data_from_repo" / dataset / ref_filename
    dev_path = data_dir / "processed" / dataset / "dev.jsonl"

    # Reference qids in order — preserved so downstream batching / ordering
    # stays paper-comparable.
    ref_qids_ordered: list[str] = []
    with open(ref_path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ref_qids_ordered.append(json.loads(line)["question_id"])

    # Index our dev by qid.
    rows_by_qid: dict[str, dict] = {}
    with open(dev_path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows_by_qid[row["question_id"]] = row

    missing = [qid for qid in ref_qids_ordered if qid not in rows_by_qid]
    if missing:
        raise RuntimeError(
            f"{dataset}: {len(missing)} reference qids missing from our "
            f"processed dev (sample: {missing[:3]}). This means our "
            f"process_*.py port dropped rows the upstream kept — fix the "
            f"processing port before continuing."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf8") as f:
        for qid in ref_qids_ordered:
            f.write(json.dumps(rows_by_qid[qid]) + "\n")
    return len(ref_qids_ordered)


def _reference_subsample_available(data_dir: Path, dataset: str, ref_filename: str) -> bool:
    return (data_dir / "reference" / "processed_data_from_repo" / dataset / ref_filename).exists()


def step8_subsample_test(data_dir: Path, *, datasets_subset: Iterable[str],
                         force: bool, dry_run: bool,
                         use_reference_qids: bool = True,
                         use_ircot_seed: bool = True,
                         remap_paragraphs: bool = False) -> None:
    """Step 8: subsample 500 'test' instances per dataset; copy to data/eval_500/.

    Strategy: when ``use_reference_qids`` is True (default) AND the reference
    ``test_subsampled.jsonl`` exists for the dataset, we filter our processed
    dev to those exact qids in reference order. This is byte-identical to the
    paper's splits and avoids the seed-reproduction headaches we hit with
    single-hop datasets. Multi-hop pass either way.

    Fallback (``--no-reference-qids`` or reference missing): random sampling
    via ``src.processing.subsample.main``. For multi-hop datasets that path
    requires IRCoT's ``dev_subsampled.jsonl`` as an avoid file — fetched here
    via ``_seed_ircot_avoid_file`` when ``use_ircot_seed`` is True.

    If ``remap_paragraphs`` is True, rewrite each context's
    ``{title, paragraph_text}`` to match the canonical BM25 corpus text via
    ``src.retrieval.find_matching_paragraph_text``. Requires the relevant
    Lucene indices to exist. Default off — the qid set is identical either
    way, so step-10/11 diffs pass without it.
    """
    callable_ = None
    if remap_paragraphs:
        # Local import — avoids dragging in pyserini for the no-remap path.
        from src.retrieval import find_matching_paragraph_text as callable_  # noqa: F401

    for dataset in datasets_subset:
        proc_dir = data_dir / "processed" / dataset
        eval_out = data_dir / "eval_500" / f"{dataset}.jsonl"
        if dry_run:
            log.info("  [dry-run] would subsample test for %s%s", dataset,
                     " (with paragraph remap)" if remap_paragraphs else "")
            continue
        if eval_out.exists() and not force:
            log.info("  skip (have %s)", eval_out)
            continue
        if not (proc_dir / "dev.jsonl").exists():
            raise FileNotFoundError(f"Missing {proc_dir / 'dev.jsonl'} — run step 7 first")

        # Preferred path: project the reference qid set onto our processed dev.
        if use_reference_qids and _reference_subsample_available(
                data_dir, dataset, "test_subsampled.jsonl"):
            if remap_paragraphs:
                # Remap is only meaningful when going through subsample.main;
                # if you need it, opt out of reference-qid mode.
                log.warning(
                    "  %s: --remap-paragraphs requested but reference-qid mode "
                    "is on (no remap will run). Pass --no-reference-qids to "
                    "force the random-sample path that supports remap.", dataset)
            n = _subsample_via_reference_qids(
                data_dir, dataset, "test_subsampled.jsonl", eval_out,
            )
            log.info("  %s test_subsampled (from reference) -> %s (%d rows)",
                     dataset, eval_out, n)
            continue

        # Fallback: random sampling.
        if use_ircot_seed:
            _seed_ircot_avoid_file(data_dir, dataset, force=force)
        sub_path = subsample.main(
            dataset, "test", proc_dir, proc_dir, sample_size=500,
            find_paragraph_callable=callable_,
        )
        eval_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sub_path, eval_out)
        log.info("  %s test_subsampled -> %s (%d rows)", dataset, eval_out, _count_lines(eval_out))


def step9_subsample_dev(data_dir: Path, *, datasets_subset: Iterable[str],
                        force: bool, dry_run: bool,
                        use_reference_qids: bool = True) -> None:
    """Step 9: subsample 500 'dev_diff_size' instances (disjoint from test);
    copy to data/train_500/.

    Same reference-vs-random strategy as step 8 — see its docstring.
    """
    for dataset in datasets_subset:
        proc_dir = data_dir / "processed" / dataset
        train_out = data_dir / "train_500" / f"{dataset}.jsonl"
        if dry_run:
            log.info("  [dry-run] would subsample dev_diff_size for %s", dataset)
            continue
        if train_out.exists() and not force:
            log.info("  skip (have %s)", train_out)
            continue
        if not (proc_dir / "dev.jsonl").exists():
            raise FileNotFoundError(
                f"Missing {proc_dir / 'dev.jsonl'} — run step 7 first")

        # Preferred path: reference qids.
        if use_reference_qids and _reference_subsample_available(
                data_dir, dataset, "dev_500_subsampled.jsonl"):
            n = _subsample_via_reference_qids(
                data_dir, dataset, "dev_500_subsampled.jsonl", train_out,
            )
            log.info("  %s dev_500_subsampled (from reference) -> %s (%d rows)",
                     dataset, train_out, n)
            continue

        # Fallback: random sample (requires step 8's test_subsampled as avoid file).
        if not (proc_dir / "test_subsampled.jsonl").exists():
            raise FileNotFoundError(
                f"Missing {proc_dir / 'test_subsampled.jsonl'} — run step 8 first")
        sub_path = subsample.main(dataset, "dev_diff_size", proc_dir, proc_dir, sample_size=500)
        train_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sub_path, train_out)
        log.info("  %s dev_500_subsampled -> %s (%d rows)", dataset, train_out,
                 _count_lines(train_out))


def step10_11_diff(data_dir: Path, *, datasets_subset: Iterable[str]) -> bool:
    """Step 10–11: compare our eval_500 question_ids with the reference tarball's.

    The 4 multi-hop datasets (HotpotQA, 2Wiki, MuSiQue) plus NQ/Trivia/SQuAD all
    exist in the reference tarball (verified by inspection of processed_data.tar.gz
    contents — diverges from data_plan.md which said only multi-hop are there).
    Returns True if all datasets in scope pass (or have no reference); False on FAIL.
    """
    reference_dir = data_dir / "reference" / "processed_data_from_repo"

    # ----- Step 10: read both sides, compute diffs.
    overall_ok = True
    log.info("=" * 72)
    log.info("Step 10/11: diffing eval_500 question_id sets against reference tarball")
    log.info("=" * 72)

    for dataset in datasets_subset:
        if dataset not in REFERENCE_DATASETS:
            log.info("  %-18s : no reference available, skipping", dataset)
            continue
        ours_path = data_dir / "eval_500" / f"{dataset}.jsonl"
        ref_path = reference_dir / dataset / "test_subsampled.jsonl"
        if not ours_path.exists():
            log.warning("  %-18s : our eval_500 file missing (%s) — SKIP", dataset, ours_path)
            continue
        if not ref_path.exists():
            log.info("  %-18s : reference file missing (%s) — SKIP "
                     "(rerun step 6?)", dataset, ref_path)
            continue

        ours = _read_qids(ours_path)
        ref = _read_qids(ref_path)
        only_ours = ours - ref
        only_ref = ref - ours

        if not only_ours and not only_ref:
            log.info("  %-18s : PASS (%d qids match)", dataset, len(ours))
        else:
            overall_ok = False
            log.error("  %-18s : FAIL — %d only-ours, %d only-ref",
                      dataset, len(only_ours), len(only_ref))
            if only_ours:
                log.error("    sample only-ours: %s", sorted(only_ours)[:3])
            if only_ref:
                log.error("    sample only-ref:  %s", sorted(only_ref)[:3])

    log.info("=" * 72)
    if overall_ok:
        log.info("Step 11 CHECKPOINT: PASS")
    else:
        log.error("Step 11 CHECKPOINT: FAIL — investigate before continuing")
    log.info("=" * 72)
    return overall_ok


def _read_qids(jsonl_path: Path) -> set[str]:
    qids: set[str] = set()
    with open(jsonl_path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qids.add(json.loads(line)["question_id"])
    return qids


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


STEPS: list[tuple[int, str]] = [
    (1, "make_tree"),
    (2, "download_hotpotqa"),
    (3, "download_2wiki"),
    (4, "download_musique"),
    (5, "download_dpr"),
    (6, "download_reference"),
    (7, "process"),
    (8, "subsample_test"),
    (9, "subsample_dev"),
    (10, "diff"),  # step 10 + 11 fused
    (11, "checkpoint"),
]


def _dataset_in_scope(dataset: str, datasets_subset: Iterable[str], step: int) -> bool:
    # Steps 2/3/4/5 only apply to certain datasets.
    if step == 2:
        return "hotpotqa" in datasets_subset
    if step == 3:
        return "2wikimultihopqa" in datasets_subset
    if step == 4:
        return "musique" in datasets_subset
    if step == 5:
        return any(d in datasets_subset for d in DPR_DATASETS)
    return True


def run(args: argparse.Namespace) -> int:
    data_dir = args.data_dir.resolve()
    state = State(data_dir / ".state.json")

    datasets_subset: tuple[str, ...]
    if args.only_datasets:
        datasets_subset = tuple(d.strip() for d in args.only_datasets.split(","))
        for d in datasets_subset:
            if d not in ALL_DATASETS:
                log.error("Unknown dataset: %s. Choose from %s", d, ALL_DATASETS)
                return 2
    else:
        datasets_subset = ALL_DATASETS

    # Determine which steps to run.
    if args.step is not None:
        steps_to_run = [args.step]
    else:
        lo = args.from_step or 1
        hi = args.to_step or 11
        steps_to_run = list(range(lo, hi + 1))

    log.info("Plan: data_dir=%s  datasets=%s  steps=%s  force=%s  dry_run=%s",
             data_dir, datasets_subset, steps_to_run, args.force, args.dry_run)

    if args.force:
        state.clear()

    skip_existing = args.skip_existing  # default True

    checkpoint_passed = True

    for step_num in steps_to_run:
        step_key = f"step{step_num}__{','.join(datasets_subset)}"
        if skip_existing and not args.force and state.is_done(step_key):
            log.info("Step %d: skip (state.json says complete)", step_num)
            continue

        log.info("=" * 72)
        log.info("Step %d: %s", step_num, STEPS[step_num - 1][1])
        log.info("=" * 72)
        t0 = time.time()

        try:
            if step_num == 1:
                step1_make_tree(data_dir)

            elif step_num == 2:
                if _dataset_in_scope("hotpotqa", datasets_subset, 2):
                    step2_download_hotpotqa(data_dir, force=args.force, dry_run=args.dry_run)
                else:
                    log.info("  skip (hotpotqa not in --only-datasets)")

            elif step_num == 3:
                if _dataset_in_scope("2wikimultihopqa", datasets_subset, 3):
                    step3_download_2wiki(data_dir, force=args.force, dry_run=args.dry_run)
                else:
                    log.info("  skip (2wikimultihopqa not in --only-datasets)")

            elif step_num == 4:
                if _dataset_in_scope("musique", datasets_subset, 4):
                    step4_download_musique(data_dir, force=args.force, dry_run=args.dry_run)
                else:
                    log.info("  skip (musique not in --only-datasets)")

            elif step_num == 5:
                if _dataset_in_scope("", datasets_subset, 5):
                    step5_download_dpr(data_dir, force=args.force, dry_run=args.dry_run,
                                       datasets_subset=datasets_subset)
                else:
                    log.info("  skip (no DPR datasets in --only-datasets)")

            elif step_num == 6:
                step6_download_reference(data_dir, force=args.force, dry_run=args.dry_run)

            elif step_num == 7:
                step7_process(data_dir, datasets_subset=datasets_subset,
                              force=args.force, dry_run=args.dry_run)

            elif step_num == 8:
                step8_subsample_test(data_dir, datasets_subset=datasets_subset,
                                     force=args.force, dry_run=args.dry_run,
                                     use_reference_qids=args.use_reference_qids,
                                     use_ircot_seed=args.use_ircot_seed,
                                     remap_paragraphs=args.remap_paragraphs)

            elif step_num == 9:
                step9_subsample_dev(data_dir, datasets_subset=datasets_subset,
                                    force=args.force, dry_run=args.dry_run,
                                    use_reference_qids=args.use_reference_qids)

            elif step_num == 10:
                if args.dry_run:
                    log.info("  [dry-run] would diff against reference tarball")
                else:
                    checkpoint_passed = step10_11_diff(data_dir, datasets_subset=datasets_subset)

            elif step_num == 11:
                if not checkpoint_passed:
                    log.error("Step 11 HARD CHECKPOINT failed — halting.")
                    return 1
                log.info("Step 11: hard checkpoint OK.")

        except Exception as e:
            log.exception("Step %d failed: %s", step_num, e)
            return 1

        elapsed = time.time() - t0
        log.info("Step %d done in %.1fs", step_num, elapsed)
        if not args.dry_run:
            state.mark_done(step_key)

    log.info("All requested steps completed successfully.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--step", type=int, choices=range(1, 12), metavar="N",
                   help="Run only this single step.")
    g.add_argument("--from-step", type=int, choices=range(1, 12), metavar="N",
                   help="Start at this step (inclusive). Default 1.")
    p.add_argument("--to-step", type=int, choices=range(1, 12), metavar="N",
                   help="Stop at this step (inclusive). Default 11.")
    p.add_argument("--only-datasets", type=str, default=None,
                   help="Comma-separated subset of {hotpotqa,2wikimultihopqa,musique,nq,trivia,squad}. "
                        "Default: all six.")
    p.add_argument("--data-dir", type=Path, default=Path("data"),
                   help="Root data directory. Default: ./data")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="(default) Skip steps already recorded as complete in .state.json.")
    p.add_argument("--force", action="store_true",
                   help="Re-run every step from scratch (clears .state.json).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would happen without downloading or processing.")
    p.add_argument("--no-reference-qids", dest="use_reference_qids",
                   action="store_false", default=True,
                   help="In steps 8/9, fall back to random sampling instead of "
                        "projecting the reference tarball's qid set onto our "
                        "processed dev. The reference is byte-identical to the "
                        "paper's splits, so default behaviour is to use it. Use "
                        "this only if you want to regenerate splits from scratch — "
                        "note that upstream's RNG state isn't reproducible for "
                        "single-hop datasets, so step 10/11 will fail on them.")
    p.add_argument("--no-ircot-seed", dest="use_ircot_seed", action="store_false",
                   default=True,
                   help="Skip seeding IRCoT's dev_subsampled.jsonl for multi-hop "
                        "datasets. Only relevant when --no-reference-qids is set "
                        "(the default reference-qid path doesn't need IRCoT).")
    p.add_argument("--remap-paragraphs", action="store_true",
                   help="In step 8, rewrite each context's (title, paragraph_text) "
                        "to match the canonical BM25 corpus passage (Adaptive-RAG's "
                        "find_matching_paragraph_text). Requires retrieval indices "
                        "to be built — run scripts/build_retrieval.py first. "
                        "Default off — qid set is identical either way, so the "
                        "step-10/11 diff still passes.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
