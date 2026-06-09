from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.annotate_lib import prompts as prompts_mod


log = logging.getLogger(__name__)


SENTENCE_ENDINGS = (".", "?", "!", "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Probe initial multi-hop route predictions and optionally downgrade them to single-step."
    )
    p.add_argument("--input-routes", type=Path, required=True)
    p.add_argument("--output-file", type=Path, required=True)
    p.add_argument("--summary-file", type=Path, default=None)
    p.add_argument("--model", default="google/gemma-4-26B-A4B-it", help="Model name exposed by the OpenAI-compatible server.")
    p.add_argument("--base-url", default="http://localhost:11434/v1")
    p.add_argument("--api-key", default="ollama")
    p.add_argument(
        "--api-mode",
        choices=("completions", "chat"),
        default="completions",
        help="Use raw completions for Adaptive-RAG completion-style prompts, or chat completions for chat servers.",
    )
    p.add_argument("--threshold", type=float, default=0.5, help="Minimum first-sentence token probability to downgrade.")
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--limit", type=int, default=None, help="Debug limit after reading input rows.")
    p.add_argument("--resume", action="store_true", help="Skip question_ids already present in --output-file.")
    p.add_argument("--start-ollama", action="store_true", help="Start `ollama serve` if the API is not reachable.")
    p.add_argument("--pull-model", action="store_true", help="Run `ollama pull MODEL` before probing.")
    p.add_argument(
        "--allow-missing-logprobs",
        action="store_false",
        dest="strict_logprobs",
        help="Debug only: keep multi_step instead of failing if token logprobs are unavailable.",
    )
    p.set_defaults(strict_logprobs=True)
    return p.parse_args()


def ollama_root(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root


def ensure_ollama(args: argparse.Namespace) -> subprocess.Popen | None:
    root = ollama_root(args.base_url)
    try:
        httpx.get(root + "/api/tags", timeout=3.0).raise_for_status()
    except Exception:
        if not args.start_ollama:
            raise RuntimeError(
                f"Ollama is not reachable at {root}. Start it first, or rerun with --start-ollama."
            )
        log.info("starting ollama serve")
        proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                httpx.get(root + "/api/tags", timeout=3.0).raise_for_status()
                break
            except Exception:
                time.sleep(1)
        else:
            proc.terminate()
            raise RuntimeError(f"Timed out waiting for Ollama at {root}")
    else:
        proc = None

    if args.pull_model:
        log.info("pulling ollama model %s", args.model)
        subprocess.run(["ollama", "pull", args.model], check=True)
    return proc


def load_route_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
    done: set[str] = set()
    with open(path, "r", encoding="utf8") as fh:
        for line in fh:
            if line.strip():
                done.add(str(json.loads(line)["question_id"]))
    return done


def build_probe_prompt(row: dict[str, Any]) -> str:
    return prompts_mod.build_prompt(row["dataset"], row["question_text"])


def first_sentence_tokens(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for token in tokens:
        out.append(token)
        text = str(token.get("token", ""))
        if any(mark in text for mark in SENTENCE_ENDINGS):
            break
    return out


def token_probability(token: dict[str, Any]) -> float | None:
    logprob = token.get("logprob")
    if logprob is None:
        return None
    try:
        return math.exp(float(logprob))
    except (TypeError, ValueError, OverflowError):
        return None


def extract_chat_tokens_with_logprobs(choice: Any) -> list[dict[str, Any]]:
    logprobs = getattr(choice, "logprobs", None)
    if logprobs is None:
        return []
    content = getattr(logprobs, "content", None)
    if content is None:
        return []
    tokens = []
    for item in content:
        token = getattr(item, "token", None)
        logprob = getattr(item, "logprob", None)
        tokens.append({"token": token, "logprob": logprob})
    return tokens


def extract_completion_tokens_with_logprobs(choice: Any) -> list[dict[str, Any]]:
    logprobs = getattr(choice, "logprobs", None)
    if logprobs is None:
        return []
    tokens = getattr(logprobs, "tokens", None) or []
    token_logprobs = getattr(logprobs, "token_logprobs", None) or []
    return [
        {"token": token, "logprob": logprob}
        for token, logprob in zip(tokens, token_logprobs)
    ]


def probe_one(client: OpenAI, args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    prompt = build_probe_prompt(row)
    start = time.perf_counter()
    if args.api_mode == "completions":
        response = client.completions.create(
            model=args.model,
            prompt=prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            logprobs=0,
            timeout=args.timeout,
        )
    else:
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            logprobs=True,
            top_logprobs=0,
            timeout=args.timeout,
        )
    latency_ms = (time.perf_counter() - start) * 1000
    choice = response.choices[0]
    if args.api_mode == "completions":
        text = choice.text or ""
        all_tokens = extract_completion_tokens_with_logprobs(choice)
    else:
        text = choice.message.content or ""
        all_tokens = extract_chat_tokens_with_logprobs(choice)
    sent_tokens = first_sentence_tokens(all_tokens)
    probs = [p for p in (token_probability(t) for t in sent_tokens) if p is not None]

    if not sent_tokens or len(probs) != len(sent_tokens):
        if args.strict_logprobs:
            raise RuntimeError(
                "The model server response did not include usable token logprobs. "
                "Use a backend/API mode that supports logprobs, such as vLLM /v1/completions."
            )
        confident = False
        min_prob = None
        mean_prob = None
    else:
        min_prob = min(probs)
        mean_prob = sum(probs) / len(probs)
        confident = all(prob >= args.threshold for prob in probs)

    final_route = "single_step" if confident else "multi_step"
    out = dict(row)
    out.update(
        {
            "probe_model": args.model,
            "probe_api_mode": args.api_mode,
            "probe_threshold": args.threshold,
            "probe_prompt": prompt,
            "probe_text": text,
            "probe_first_sentence_tokens": sent_tokens,
            "probe_first_sentence_min_prob": min_prob,
            "probe_first_sentence_mean_prob": mean_prob,
            "probe_confident": confident,
            "probe_latency_ms": latency_ms,
            "final_route": final_route,
            "final_route_source": "confidence_probe",
            "final_route_changed": final_route != row.get("initial_route"),
        }
    )
    return out


def passthrough_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out.update(
        {
            "probe_model": None,
            "probe_api_mode": None,
            "probe_threshold": None,
            "probe_text": None,
            "probe_first_sentence_tokens": [],
            "probe_first_sentence_min_prob": None,
            "probe_first_sentence_mean_prob": None,
            "probe_confident": None,
            "probe_latency_ms": 0.0,
            "final_route": row["initial_route"],
            "final_route_source": "initial_classifier",
            "final_route_changed": False,
        }
    )
    return out


def write_summary(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    route_counts: dict[str, int] = {}
    final_counts: dict[str, int] = {}
    probed = 0
    downgraded = 0
    probe_latencies = []
    for row in rows:
        route_counts[row["initial_route"]] = route_counts.get(row["initial_route"], 0) + 1
        final_counts[row["final_route"]] = final_counts.get(row["final_route"], 0) + 1
        if row["final_route_source"] == "confidence_probe":
            probed += 1
            probe_latencies.append(float(row["probe_latency_ms"]))
            if row["final_route_changed"]:
                downgraded += 1
    summary = {
        "input_routes": str(args.input_routes),
        "output_file": str(args.output_file),
        "probe_model": args.model,
        "probe_api_mode": args.api_mode,
        "threshold": args.threshold,
        "num_rows": len(rows),
        "num_probed": probed,
        "num_downgraded_multi_to_single": downgraded,
        "initial_route_counts": route_counts,
        "final_route_counts": final_counts,
        "probe_latency": {
            "total_ms": sum(probe_latencies),
            "mean_ms": sum(probe_latencies) / len(probe_latencies) if probe_latencies else 0.0,
            "min_ms": min(probe_latencies) if probe_latencies else 0.0,
            "max_ms": max(probe_latencies) if probe_latencies else 0.0,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf8")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary_file = args.summary_file or args.output_file.with_suffix(".summary.json")
    ollama_proc = ensure_ollama(args) if (args.start_ollama or args.pull_model) else None
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    rows = load_route_rows(args.input_routes, args.limit)
    done_ids = load_done_ids(args.output_file) if args.resume else set()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    written_rows: list[dict[str, Any]] = []

    try:
        with open(args.output_file, mode, encoding="utf8") as fh:
            for row in tqdm(rows, desc="probe multi routes"):
                qid = str(row["question_id"])
                if qid in done_ids:
                    continue
                if row.get("initial_route") == "multi_step":
                    out = probe_one(client, args, row)
                else:
                    out = passthrough_row(row)
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
                fh.flush()
                written_rows.append(out)
    finally:
        if ollama_proc is not None:
            ollama_proc.terminate()

    all_output_rows = load_route_rows(args.output_file, None)
    write_summary(summary_file, all_output_rows, args)
    log.info("wrote %d new rows; total output rows=%d", len(written_rows), len(all_output_rows))


if __name__ == "__main__":
    main()
