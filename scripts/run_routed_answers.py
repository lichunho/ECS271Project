from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import httpx
from openai import OpenAI
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.annotate_lib import extract, normalise, prompts as prompts_mod
from src.annotate_lib.llm_adapter import estimate_tokens

log = logging.getLogger("run_routed_answers")

CONTEXT_CHAR_CAP = 24_000
IRCOT_MAX_TOKENS_PER_STEP = 200
DEFAULT_MAX_TOKENS = 200
DEFAULT_TEMPERATURE = 0.0
SENT_SPLIT = __import__("re").compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Passage:
    doc_id: str
    title: str
    text: str
    score: float = 0.0


@dataclass
class AnswerAttempt:
    strategy: str
    pred_raw: str
    pred_extracted: str
    em: int
    f1: float
    ok: bool
    latency_s: float
    llm_latency_s: float
    retrieval_latency_s: float
    prompt_tokens_est: int
    completion_tokens_est: int
    context_doc_ids: list[str] = field(default_factory=list)
    n_hops: int = 0
    error: str = ""

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["f1"] = round(float(d["f1"]), 4)
        d["latency_s"] = round(float(d["latency_s"]), 3)
        d["llm_latency_s"] = round(float(d["llm_latency_s"]), 3)
        d["retrieval_latency_s"] = round(float(d["retrieval_latency_s"]), 3)
        return d


class Retriever:
    def search(self, dataset: str, query: str, k: int) -> list[Passage]:
        raise NotImplementedError


class NullRetriever(Retriever):
    def search(self, dataset: str, query: str, k: int) -> list[Passage]:
        return []


class PyseriniRetriever(Retriever):
    def __init__(self, index_root: str | None = None) -> None:
        from src.retrieval import get_retriever
        self.get_retriever = get_retriever
        self.index_root = index_root
        self._cache: dict[str, Any] = {}

    def search(self, dataset: str, query: str, k: int) -> list[Passage]:
        if dataset not in self._cache:
            self._cache[dataset] = self.get_retriever(dataset, self.index_root)
        return self._cache[dataset].search(query, k=k)


class AdaptiveHttpRetriever(Retriever):
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def search(self, dataset: str, query: str, k: int) -> list[Passage]:
        payload = {
            "retrieval_method": "retrieve_from_elasticsearch",
            "query_text": query,
            "corpus_name": es_corpus_for_dataset(dataset),
            "max_hits_count": k,
            "max_buffer_count": max(100, k),
        }
        resp = self.client.post(f"{self.base_url}/retrieve/", json=payload)
        resp.raise_for_status()
        return passages_from_sources(resp.json().get("retrieval") or [])


class ElasticsearchRetriever(Retriever):
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def search(self, dataset: str, query: str, k: int) -> list[Passage]:
        body = {
            "size": max(100, k),
            "_source": ["id", "title", "paragraph_text", "url", "is_abstract", "paragraph_index"],
            "query": {"bool": {"should": [{"match": {"paragraph_text": query}}]}},
        }
        corpus = es_corpus_for_dataset(dataset)
        resp = self.client.post(f"{self.base_url}/{corpus}/_search", json=body)
        resp.raise_for_status()
        hits = resp.json().get("hits", {}).get("hits", [])
        sources = []
        seen_text: set[str] = set()
        for hit in hits:
            src = dict(hit.get("_source") or {})
            text = (src.get("paragraph_text") or "").strip().lower()
            if text in seen_text:
                continue
            seen_text.add(text)
            src["score"] = hit.get("_score", 0.0)
            src["corpus_name"] = dataset
            sources.append(src)
            if len(sources) >= k:
                break
        return passages_from_sources(sources)


def es_corpus_for_dataset(dataset: str) -> str:
    if dataset in {"nq", "trivia", "squad"}:
        return "wiki"
    return dataset


def passages_from_sources(items: Iterable[dict[str, Any]]) -> list[Passage]:
    out: list[Passage] = []
    for item in items:
        text = item.get("paragraph_text") or item.get("text") or item.get("contents") or ""
        title = item.get("title") or ""
        doc_id = item.get("id") or item.get("doc_id") or item.get("_id") or ""
        score = item.get("score") or item.get("_score") or 0.0
        out.append(Passage(doc_id=str(doc_id), title=str(title), text=str(text), score=float(score or 0.0)))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run final routed QA answers from probed route decisions.")
    p.add_argument("--input-file", type=Path, required=True, help="*.probed.jsonl or *.routes.jsonl with final_route/initial_route.")
    p.add_argument("--output-file", type=Path, required=True)
    p.add_argument("--summary-file", type=Path, default=None)
    p.add_argument("--model", default="google/gemma-4-26b-a4b")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--api-mode", choices=("completions", "chat"), default="completions")
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--retriever-backend", choices=("none", "pyserini", "adaptive-http", "elasticsearch"), default="elasticsearch")
    p.add_argument("--retriever-url", default="http://localhost:9200", help="ES root for elasticsearch, retriever server root for adaptive-http.")
    p.add_argument("--index-root", default=None, help="Optional Pyserini index root.")
    p.add_argument("--bm25-k", type=int, default=15)
    p.add_argument("--ircot-max-iters", type=int, default=4)
    p.add_argument("--ircot-k-per-step", type=int, default=6)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--include-prompts", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    return p.parse_args()


def load_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    with open(path, "r", encoding="utf8") as fh:
        for line in fh:
            if line.strip():
                out.add(str(json.loads(line).get("question_id")))
    return out


def gold_aliases(row: dict[str, Any]) -> list[str]:
    out = []
    for obj in row.get("answers_objects", []) or []:
        for span in obj.get("spans", []) or []:
            if span:
                out.append(str(span))
        if obj.get("number"):
            out.append(str(obj["number"]))
        date = obj.get("date") or {}
        if isinstance(date, dict):
            stamp = " ".join(str(date.get(k, "")) for k in ("day", "month", "year") if date.get(k, ""))
            if stamp.strip():
                out.append(stamp.strip())
    return out


def score_answer(pred: str, row: dict[str, Any]) -> tuple[int, float, bool]:
    aliases = gold_aliases(row)
    em_v = normalise.em(pred, aliases)
    f1_v = normalise.f1(pred, aliases)
    return em_v, f1_v, bool(em_v)


def truncate_passages(passages: list[Passage], char_cap: int = CONTEXT_CHAR_CAP) -> list[Passage]:
    out = []
    used = 0
    for p in passages:
        approx = len(p.text or "") + len(p.title or "") + 20
        if used + approx > char_cap and out:
            break
        out.append(p)
        used += approx
    return out


def completion_text_and_usage(response: Any, api_mode: str) -> tuple[str, int | None, int | None]:
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
    choice = response.choices[0]
    if api_mode == "completions":
        return choice.text or "", prompt_tokens, completion_tokens
    return choice.message.content or "", prompt_tokens, completion_tokens


def llm_complete(client: OpenAI, args: argparse.Namespace, prompt: str, *, max_tokens: int | None = None) -> tuple[str, float, int, int]:
    t0 = time.perf_counter()
    if args.api_mode == "completions":
        response = client.completions.create(
            model=args.model,
            prompt=prompt,
            temperature=args.temperature,
            max_tokens=max_tokens or args.max_tokens,
            timeout=args.timeout,
        )
    else:
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.temperature,
            max_tokens=max_tokens or args.max_tokens,
            timeout=args.timeout,
        )
    latency = time.perf_counter() - t0
    text, pt, ct = completion_text_and_usage(response, args.api_mode)
    return text, latency, pt if pt is not None else estimate_tokens(prompt), ct if ct is not None else estimate_tokens(text)


def make_attempt(strategy: str, row: dict[str, Any], raw: str, latency_s: float, llm_latency_s: float, retrieval_latency_s: float, prompt_tokens: int, completion_tokens: int, context_doc_ids: list[str], n_hops: int, error: str = "") -> AnswerAttempt:
    cot = prompts_mod.is_cot(row["dataset"]) if strategy != "multi_step" else True
    extracted = extract.extract(raw, cot=cot) if strategy != "multi_step" else extract.extract_cot(raw)
    em_v, f1_v, ok = score_answer(extracted, row)
    return AnswerAttempt(
        strategy=strategy,
        pred_raw=raw,
        pred_extracted=extracted,
        em=em_v,
        f1=f1_v,
        ok=ok,
        latency_s=latency_s,
        llm_latency_s=llm_latency_s,
        retrieval_latency_s=retrieval_latency_s,
        prompt_tokens_est=prompt_tokens,
        completion_tokens_est=completion_tokens,
        context_doc_ids=context_doc_ids,
        n_hops=n_hops,
        error=error,
    )


def answer_no_retrieval(client: OpenAI, args: argparse.Namespace, row: dict[str, Any]) -> AnswerAttempt:
    t0 = time.perf_counter()
    prompt = prompts_mod.build_prompt(row["dataset"], row["question_text"])
    raw, llm_s, pt, ct = llm_complete(client, args, prompt)
    return make_attempt("no_retrieval", row, raw, time.perf_counter() - t0, llm_s, 0.0, pt, ct, [], 0)


def answer_single_step(client: OpenAI, args: argparse.Namespace, row: dict[str, Any], retriever: Retriever) -> AnswerAttempt:
    t0 = time.perf_counter()
    rt0 = time.perf_counter()
    passages = retriever.search(row["dataset"], row["question_text"], args.bm25_k)
    retrieval_s = time.perf_counter() - rt0
    passages = truncate_passages(passages)
    prompt = prompts_mod.build_prompt(row["dataset"], row["question_text"], context_text=prompts_mod.format_context(passages))
    raw, llm_s, pt, ct = llm_complete(client, args, prompt)
    return make_attempt("single_step", row, raw, time.perf_counter() - t0, llm_s, retrieval_s, pt, ct, [p.doc_id for p in passages], 0)


def next_query_from_generation(new_chunk: str) -> str:
    chunk = new_chunk.strip()
    if not chunk:
        return ""
    parts = [p.strip() for p in SENT_SPLIT.split(chunk) if p.strip()]
    return parts[-1] if parts else chunk


def answer_multi_step(client: OpenAI, args: argparse.Namespace, row: dict[str, Any], retriever: Retriever) -> AnswerAttempt:
    t0 = time.perf_counter()
    dataset = row["dataset"]
    question = row["question_text"]
    cot_dataset = dataset if dataset in prompts_mod.COT_DATASETS else "hotpotqa"
    accumulated: list[Passage] = []
    seen: set[str] = set()
    generation_so_far = ""
    total_llm = 0.0
    total_retrieval = 0.0
    total_pt = 0
    total_ct = 0
    last_raw = ""
    n_hops = 0
    error = ""

    try:
        rt0 = time.perf_counter()
        hits = retriever.search(dataset, question, args.ircot_k_per_step)
        total_retrieval += time.perf_counter() - rt0
        for p in hits:
            if p.doc_id not in seen:
                accumulated.append(p)
                seen.add(p.doc_id)

        for hop in range(args.ircot_max_iters):
            n_hops = hop + 1
            passages = truncate_passages(accumulated)
            prompt = prompts_mod.build_prompt(
                cot_dataset,
                question,
                context_text=prompts_mod.format_context(passages),
                generation_so_far=generation_so_far,
            )
            raw, llm_s, pt, ct = llm_complete(client, args, prompt, max_tokens=IRCOT_MAX_TOKENS_PER_STEP)
            last_raw = raw
            total_llm += llm_s
            total_pt += pt
            total_ct += ct
            new_chunk = raw.strip()
            generation_so_far = (generation_so_far + " " + new_chunk).strip() if generation_so_far else new_chunk
            if extract.has_answer(generation_so_far):
                break
            next_q = next_query_from_generation(new_chunk)
            if not next_q:
                break
            rt0 = time.perf_counter()
            new_hits = retriever.search(dataset, next_q, args.ircot_k_per_step)
            total_retrieval += time.perf_counter() - rt0
            added = 0
            for p in new_hits:
                if p.doc_id not in seen:
                    accumulated.append(p)
                    seen.add(p.doc_id)
                    added += 1
            if added == 0:
                break
            if sum(len(p.text or "") + 20 for p in accumulated) >= CONTEXT_CHAR_CAP:
                break
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        if args.fail_fast:
            raise

    return make_attempt(
        "multi_step",
        row,
        generation_so_far or last_raw,
        time.perf_counter() - t0,
        total_llm,
        total_retrieval,
        total_pt,
        total_ct,
        [p.doc_id for p in accumulated],
        n_hops,
        error,
    )


def make_retriever(args: argparse.Namespace) -> Retriever:
    if args.retriever_backend == "none":
        return NullRetriever()
    if args.retriever_backend == "pyserini":
        return PyseriniRetriever(args.index_root)
    if args.retriever_backend == "adaptive-http":
        return AdaptiveHttpRetriever(args.retriever_url, args.timeout)
    if args.retriever_backend == "elasticsearch":
        return ElasticsearchRetriever(args.retriever_url, args.timeout)
    raise ValueError(args.retriever_backend)


def route_for(row: dict[str, Any]) -> str:
    route = row.get("final_route") or row.get("initial_route")
    if route not in {"no_retrieval", "single_step", "multi_step"}:
        raise ValueError(f"Bad route for qid={row.get('question_id')}: {route!r}")
    return route


def reference_route_for(row: dict[str, Any]) -> str | None:
    for key in ("oracle_label", "silver_route", "silver_label", "reference_route", "gold_route"):
        value = row.get(key)
        if value in {"no_retrieval", "single_step", "multi_step"}:
            return str(value)
    return None


def route_confusion_key(chosen: str, reference: str | None) -> str | None:
    if reference is None:
        return None
    return f"{reference}->{chosen}"


def answer_one(client: OpenAI, args: argparse.Namespace, row: dict[str, Any], retriever: Retriever) -> dict[str, Any]:
    row = dict(row)
    row.setdefault("dataset", row.get("source_dataset") or "")
    route = route_for(row)
    if route == "no_retrieval":
        attempt = answer_no_retrieval(client, args, row)
    elif route == "single_step":
        attempt = answer_single_step(client, args, row, retriever)
    else:
        attempt = answer_multi_step(client, args, row, retriever)

    reference_route = reference_route_for(row)
    classifier_ms = float(row.get("classifier_inference_ms") or 0.0)
    probe_ms = float(row.get("probe_latency_ms") or 0.0)
    answer_ms = attempt.latency_s * 1000
    out = dict(row)
    out.update(
        {
            "answer_model": args.model,
            "answer_api_mode": args.api_mode,
            "retriever_backend": args.retriever_backend,
            "executed_route": route,
            "reference_route": reference_route,
            "route_matches_reference": (route == reference_route) if reference_route is not None else None,
            "route_confusion": route_confusion_key(route, reference_route),
            "answer_attempt": attempt.to_json(),
            "answer_latency_ms": answer_ms,
            "total_pipeline_latency_ms": classifier_ms + probe_ms + answer_ms,
        }
    )
    return out


def write_summary(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    route_counts: dict[str, int] = {}
    ok_by_route: dict[str, int] = {}
    total_latency = []
    answer_latency = []
    retrieval_latency = []
    llm_latency = []
    route_reference_total = 0
    route_reference_correct = 0
    route_confusion: dict[str, int] = {}
    answer_em_by_reference_match = {"matched": {"ok": 0, "total": 0}, "mismatched": {"ok": 0, "total": 0}}
    for row in rows:
        route = row.get("executed_route") or row.get("final_route") or row.get("initial_route")
        route_counts[route] = route_counts.get(route, 0) + 1
        att = row.get("answer_attempt") or {}
        if att.get("ok"):
            ok_by_route[route] = ok_by_route.get(route, 0) + 1
        reference = row.get("reference_route")
        if reference is not None:
            route_reference_total += 1
            matched = row.get("route_matches_reference") is True
            if matched:
                route_reference_correct += 1
            key = row.get("route_confusion") or f"{reference}->{route}"
            route_confusion[key] = route_confusion.get(key, 0) + 1
            bucket = "matched" if matched else "mismatched"
            answer_em_by_reference_match[bucket]["total"] += 1
            if att.get("ok"):
                answer_em_by_reference_match[bucket]["ok"] += 1
        total_latency.append(float(row.get("total_pipeline_latency_ms") or 0.0))
        answer_latency.append(float(row.get("answer_latency_ms") or 0.0))
        retrieval_latency.append(float(att.get("retrieval_latency_s") or 0.0) * 1000)
        llm_latency.append(float(att.get("llm_latency_s") or 0.0) * 1000)

    def stats(values: list[float]) -> dict[str, float]:
        return {
            "total_ms": sum(values),
            "mean_ms": sum(values) / len(values) if values else 0.0,
            "min_ms": min(values) if values else 0.0,
            "max_ms": max(values) if values else 0.0,
        }

    summary = {
        "input_file": str(args.input_file),
        "output_file": str(args.output_file),
        "answer_model": args.model,
        "retriever_backend": args.retriever_backend,
        "num_rows": len(rows),
        "route_counts": route_counts,
        "route_reference_agreement": {
            "num_with_reference": route_reference_total,
            "num_matching_reference": route_reference_correct,
            "accuracy": route_reference_correct / route_reference_total if route_reference_total else None,
            "confusion": route_confusion,
        },
        "answer_exact_match_by_route_reference_match": {
            k: {
                "exact_match": v["ok"] / v["total"] if v["total"] else None,
                "ok": v["ok"],
                "total": v["total"],
            }
            for k, v in answer_em_by_reference_match.items()
        },
        "exact_match_by_route": ok_by_route,
        "overall_exact_match": sum(ok_by_route.values()) / len(rows) if rows else 0.0,
        "latency": {
            "total_pipeline": stats(total_latency),
            "answer": stats(answer_latency),
            "retrieval": stats(retrieval_latency),
            "llm": stats(llm_latency),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf8")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = load_rows(args.input_file, args.limit)
    done = load_done_ids(args.output_file) if args.resume else set()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file = args.summary_file or args.output_file.with_suffix(".summary.json")
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    retriever = make_retriever(args)

    mode = "a" if args.resume else "w"
    wrote = 0
    with open(args.output_file, mode, encoding="utf8") as fh:
        for row in tqdm(rows, desc="run routed answers"):
            qid = str(row.get("question_id"))
            if qid in done:
                continue
            try:
                out = answer_one(client, args, row, retriever)
            except Exception as e:
                if args.fail_fast:
                    raise
                out = dict(row)
                out.update({"executed_route": row.get("final_route") or row.get("initial_route"), "answer_error": f"{type(e).__name__}: {e}"})
            if not args.include_prompts:
                out.pop("probe_prompt", None)
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            fh.flush()
            wrote += 1

    all_rows = load_rows(args.output_file)
    write_summary(summary_file, all_rows, args)
    log.info("wrote %d new rows; total output rows=%d", wrote, len(all_rows))


if __name__ == "__main__":
    main()
