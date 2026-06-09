from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import CACHE_DIR, LABEL_MAP


log = logging.getLogger(__name__)

OPTION_TO_LABEL_ID = {"A": 0, "B": 1, "C": 2}


class QuestionOnlyDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int, source_prefix: str = "") -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.source_prefix = source_prefix

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        encoded = self.tokenizer(
            self.source_prefix + row["question_text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "row_idx": idx,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict initial Adaptive-RAG routes for a test/eval split.")
    p.add_argument("--data-dir", type=Path, default=Path("data/eval_500"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--classifier-name", required=True, help="Short name written into each row and output filename.")
    p.add_argument("--model-path", required=True, help="HF model id or local/Drive checkpoint directory.")
    p.add_argument("--model-kind", choices=("encoder", "t5"), default="encoder")
    p.add_argument("--datasets", nargs="*", default=None, help="Optional dataset names to run. Defaults to all JSONL files.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--source-prefix", type=str, default="", help="Prefix for T5-style text-to-text inputs.")
    p.add_argument("--fp16", action="store_true", help="Run inference with float16 autocast on CUDA.")
    p.add_argument("--bf16", action="store_true", help="Run inference with bfloat16 autocast on CUDA.")
    p.add_argument("--include-answers", action="store_true", help="Copy answers_objects into output rows for evaluation convenience.")
    p.add_argument("--max-samples", type=int, default=None, help="Debug limit after loading/ordering rows.")
    return p.parse_args()


def load_rows(data_dir: Path, datasets: list[str] | None, max_samples: int | None) -> list[dict[str, Any]]:
    if datasets:
        paths = [data_dir / f"{name}.jsonl" for name in datasets]
    else:
        paths = sorted(data_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No JSONL files found in {data_dir}")

    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_source_file"] = str(path)
                rows.append(row)
                if max_samples is not None and len(rows) >= max_samples:
                    return rows
    return rows


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    }


def precision_context(args: argparse.Namespace, device: torch.device):
    enabled = device.type == "cuda" and (args.fp16 or args.bf16)
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    return torch.amp.autocast("cuda", enabled=enabled, dtype=dtype)


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def predict_encoder(args: argparse.Namespace, tokenizer: Any, rows: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        cache_dir=CACHE_DIR,
    )
    model.to(device)
    model.eval()

    dataset = QuestionOnlyDataset(rows, tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size)
    predictions: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"predict {args.classifier_name}"):
            model_batch = batch_to_device(batch, device)
            synchronize_if_cuda(device)
            start = time.perf_counter()
            with precision_context(args, device):
                logits = model(**model_batch).logits
            synchronize_if_cuda(device)
            batch_inference_ms = (time.perf_counter() - start) * 1000
            per_example_ms = batch_inference_ms / max(1, int(batch["row_idx"].numel()))
            probs = torch.softmax(logits.float(), dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)
            for row_idx, pred_id, prob_vec, logit_vec in zip(
                batch["row_idx"].tolist(),
                pred_ids.cpu().tolist(),
                probs.cpu().tolist(),
                logits.float().cpu().tolist(),
            ):
                predictions.append(
                    make_output_row(
                        args,
                        rows[row_idx],
                        pred_id,
                        prob_vec,
                        logit_vec,
                        classifier_inference_ms=per_example_ms,
                        classifier_batch_size=int(batch["row_idx"].numel()),
                    )
                )

    return predictions


def predict_t5(args: argparse.Namespace, tokenizer: Any, rows: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_path,
        cache_dir=CACHE_DIR,
    )
    model.to(device)
    model.eval()

    option_token_ids = torch.tensor(
        [tokenizer(option).input_ids[0] for option in ("A", "B", "C")],
        dtype=torch.long,
        device=device,
    )
    dataset = QuestionOnlyDataset(rows, tokenizer, args.max_length, source_prefix=args.source_prefix)
    loader = DataLoader(dataset, batch_size=args.batch_size)
    predictions: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"predict {args.classifier_name}"):
            model_batch = batch_to_device(batch, device)
            synchronize_if_cuda(device)
            start = time.perf_counter()
            with precision_context(args, device):
                generated = model.generate(
                    **model_batch,
                    max_length=2,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            synchronize_if_cuda(device)
            batch_inference_ms = (time.perf_counter() - start) * 1000
            per_example_ms = batch_inference_ms / max(1, int(batch["row_idx"].numel()))
            option_scores = torch.index_select(generated.scores[0].float(), dim=1, index=option_token_ids)
            probs = torch.softmax(option_scores, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)
            for row_idx, pred_id, prob_vec, score_vec in zip(
                batch["row_idx"].tolist(),
                pred_ids.cpu().tolist(),
                probs.cpu().tolist(),
                option_scores.cpu().tolist(),
            ):
                predictions.append(
                    make_output_row(
                        args,
                        rows[row_idx],
                        pred_id,
                        prob_vec,
                        score_vec,
                        classifier_inference_ms=per_example_ms,
                        classifier_batch_size=int(batch["row_idx"].numel()),
                    )
                )

    return predictions


def make_output_row(
    args: argparse.Namespace,
    row: dict[str, Any],
    pred_id: int,
    probs: list[float],
    scores: list[float],
    *,
    classifier_inference_ms: float,
    classifier_batch_size: int,
) -> dict[str, Any]:
    out = {
        "question_id": str(row["question_id"]),
        "dataset": row["dataset"],
        "question_text": row["question_text"],
        "classifier_name": args.classifier_name,
        "classifier_model_path": args.model_path,
        "classifier_model_kind": args.model_kind,
        "initial_route_id": pred_id,
        "initial_route": LABEL_MAP[pred_id],
        "route_probabilities": {
            LABEL_MAP[i]: probs[i]
            for i in range(len(probs))
        },
        "route_scores": {
            LABEL_MAP[i]: scores[i]
            for i in range(len(scores))
        },
        "classifier_inference_ms": classifier_inference_ms,
        "classifier_batch_size": classifier_batch_size,
        "source_file": row.get("_source_file"),
    }
    if args.include_answers:
        out["answers_objects"] = row.get("answers_objects")
    return out


def write_outputs(args: argparse.Namespace, predictions: list[dict[str, Any]]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / f"{args.classifier_name}.routes.jsonl"
    with open(output_file, "w", encoding="utf8") as fh:
        for row in predictions:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts: dict[str, dict[str, int]] = {}
    latencies = [float(row["classifier_inference_ms"]) for row in predictions]
    for row in predictions:
        dataset = row["dataset"]
        route = row["initial_route"]
        counts.setdefault(dataset, {})
        counts[dataset][route] = counts[dataset].get(route, 0) + 1

    summary = {
        "classifier_name": args.classifier_name,
        "classifier_model_path": args.model_path,
        "classifier_model_kind": args.model_kind,
        "data_dir": str(args.data_dir),
        "output_file": str(output_file),
        "num_rows": len(predictions),
        "counts_by_dataset": counts,
        "latency": {
            "classifier_inference_total_ms": sum(latencies),
            "classifier_inference_mean_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "classifier_inference_min_ms": min(latencies) if latencies else 0.0,
            "classifier_inference_max_ms": max(latencies) if latencies else 0.0,
            "note": "Per-example GPU inference time only; excludes model loading and most tokenization/file I/O.",
        },
    }
    with open(args.output_dir / f"{args.classifier_name}.summary.json", "w", encoding="utf8") as fh:
        json.dump(summary, fh, indent=2)
    log.info("wrote %d route predictions to %s", len(predictions), output_file)


def main() -> None:
    args = parse_args()
    if args.fp16 and args.bf16:
        raise ValueError("Use only one of --fp16 or --bf16.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rows = load_rows(args.data_dir, args.datasets, args.max_samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s rows=%d model=%s kind=%s", device, len(rows), args.model_path, args.model_kind)

    run_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=CACHE_DIR)
    if args.model_kind == "encoder":
        predictions = predict_encoder(args, tokenizer, rows, device)
    else:
        predictions = predict_t5(args, tokenizer, rows, device)
    total_wall_ms = (time.perf_counter() - run_start) * 1000
    for row in predictions:
        row["classifier_run_total_wall_ms"] = total_wall_ms
    write_outputs(args, predictions)


if __name__ == "__main__":
    main()
