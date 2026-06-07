from __future__ import annotations

import argparse
import collections
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

from src.config import CACHE_DIR, LABEL_MAP, NUM_LABELS


log = logging.getLogger(__name__)

LABEL_TO_ID = {v: k for k, v in LABEL_MAP.items()}
ID_TO_OPTION = {0: "A", 1: "B", 2: "C"}
OPTION_TO_ID = {v: k for k, v in ID_TO_OPTION.items()}


@dataclass(frozen=True)
class Example:
    qid: str
    question: str
    label: int
    dataset: str


class T5QuestionDataset(Dataset):
    def __init__(
        self,
        examples: list[Example],
        tokenizer: Any,
        *,
        max_length: int,
        max_target_length: int,
        source_prefix: str,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_target_length = max_target_length
        self.source_prefix = source_prefix

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        encoded = self.tokenizer(
            self.source_prefix + ex.question,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer(
            ID_TO_OPTION[ex.label],
            truncation=True,
            max_length=self.max_target_length,
            padding="max_length",
            return_tensors="pt",
        )
        labels = target["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels,
            "class_labels": torch.tensor(ex.label, dtype=torch.long),
            "qid": ex.qid,
            "dataset": ex.dataset,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune T5 as an Adaptive-RAG A/B/C strategy classifier.")
    p.add_argument("--train-file", type=Path, default=Path("data/labeled/classifier_train_binary_silver.jsonl"))
    p.add_argument("--validation-file", type=Path, default=Path("data/labeled/classifier_valid_silver.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/classifier/t5-large-binary-silver"))
    p.add_argument("--model-name", type=str, default="t5-large")
    p.add_argument("--source-prefix", type=str, default="")
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--max-target-length", type=int, default=2)
    p.add_argument("--generation-max-length", type=int, default=2)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=13370)
    p.add_argument(
        "--selection-metric",
        choices=("accuracy", "balanced_accuracy", "macro_f1"),
        default="accuracy",
        help="Validation metric used to choose the saved best model.",
    )
    p.add_argument("--fp16", action="store_true", help="Use CUDA automatic mixed precision.")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--no-epoch-checkpoints", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_examples(path: Path, max_samples: int | None = None) -> list[Example]:
    examples: list[Example] = []
    with open(path, "r", encoding="utf8") as fh:
        for line in fh:
            row = json.loads(line)
            examples.append(
                Example(
                    qid=str(row["question_id"]),
                    question=row["question_text"],
                    label=LABEL_TO_ID[row["oracle_label"]],
                    dataset=row["dataset"],
                )
            )
            if max_samples is not None and len(examples) >= max_samples:
                break
    return examples


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(device),
    }


def compute_metrics(
    *,
    labels: list[int],
    preds: list[int],
    datasets: list[str],
) -> dict[str, Any]:
    correct = 0
    total = len(labels)
    per_label: dict[str, dict[str, int]] = {
        name: {"correct": 0, "total": 0, "predicted": 0}
        for name in LABEL_MAP.values()
    }
    per_dataset: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"correct": 0, "total": 0})

    for pred, label, dataset in zip(preds, labels, datasets):
        ok = pred == label
        label_name = LABEL_MAP[label]
        pred_name = LABEL_MAP[pred]
        per_label[label_name]["total"] += 1
        per_label[pred_name]["predicted"] += 1
        per_dataset[dataset]["total"] += 1
        if ok:
            correct += 1
            per_label[label_name]["correct"] += 1
            per_dataset[dataset]["correct"] += 1

    metrics: dict[str, Any] = {"accuracy": correct / total if total else 0.0}
    metrics["per_label"] = {
        label: {
            **values,
            "accuracy": values["correct"] / values["total"] if values["total"] else 0.0,
        }
        for label, values in per_label.items()
    }
    recalls = [values["accuracy"] for values in metrics["per_label"].values()]
    metrics["balanced_accuracy"] = sum(recalls) / len(recalls) if recalls else 0.0
    f1s = []
    for values in metrics["per_label"].values():
        precision = values["correct"] / values["predicted"] if values["predicted"] else 0.0
        recall = values["accuracy"]
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        values["precision"] = precision
        values["f1"] = f1
        f1s.append(f1)
    metrics["macro_f1"] = sum(f1s) / len(f1s) if f1s else 0.0
    metrics["per_dataset"] = {
        dataset: {
            **values,
            "accuracy": values["correct"] / values["total"] if values["total"] else 0.0,
        }
        for dataset, values in sorted(per_dataset.items())
    }
    return metrics


def evaluate(
    model: AutoModelForSeq2SeqLM,
    tokenizer: Any,
    dataloader: DataLoader,
    device: torch.device,
    *,
    generation_max_length: int,
) -> dict[str, Any]:
    model.eval()
    option_token_ids = torch.tensor(
        [tokenizer(option).input_ids[0] for option in ("A", "B", "C")],
        dtype=torch.long,
        device=device,
    )
    labels: list[int] = []
    preds: list[int] = []
    datasets: list[str] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=generation_max_length,
                return_dict_in_generate=True,
                output_scores=True,
            )
            first_step_scores = generated.scores[0]
            option_scores = torch.index_select(first_step_scores, dim=1, index=option_token_ids)
            batch_preds = torch.argmax(option_scores, dim=-1).cpu().tolist()
            preds.extend(batch_preds)
            labels.extend(batch["class_labels"].tolist())
            datasets.extend(batch["dataset"])

    return compute_metrics(labels=labels, preds=preds, datasets=datasets)


def save_epoch_checkpoint(
    *,
    checkpoint_dir: Path,
    model: AutoModelForSeq2SeqLM,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_score: float,
    history: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save(
        {
            "epoch": epoch,
            "best_score": best_score,
            "history": history,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
        checkpoint_dir / "training_state.pt",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output_dir / "train.log", mode="w", encoding="utf8"),
        ],
    )
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s model=%s output_dir=%s", device, args.model_name, args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=CACHE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, cache_dir=CACHE_DIR)
    model.to(device)
    if args.fp16:
        model.float()

    train_examples = load_examples(args.train_file, args.max_train_samples)
    eval_examples = load_examples(args.validation_file, args.max_eval_samples)
    log.info("train=%d valid=%d", len(train_examples), len(eval_examples))
    log.info("train labels=%s", collections.Counter(ex.label for ex in train_examples))
    log.info("valid labels=%s", collections.Counter(ex.label for ex in eval_examples))

    train_dataset = T5QuestionDataset(
        train_examples,
        tokenizer,
        max_length=args.max_length,
        max_target_length=args.max_target_length,
        source_prefix=args.source_prefix,
    )
    eval_dataset = T5QuestionDataset(
        eval_examples,
        tokenizer,
        max_length=args.max_length,
        max_target_length=args.max_target_length,
        source_prefix=args.source_prefix,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    update_steps_per_epoch = max(1, (len(train_loader) + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps)
    total_steps = args.epochs * update_steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")

    log.info(
        "epochs=%d lr=%g batch_size=%d grad_accum=%d effective_batch_size=%d max_length=%d selection_metric=%s",
        args.epochs,
        args.learning_rate,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.batch_size * args.gradient_accumulation_steps,
        args.max_length,
        args.selection_metric,
    )

    best_score = -1.0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(progress, start=1):
            model_batch = batch_to_device(batch, device)
            with torch.amp.autocast("cuda", enabled=args.fp16 and device.type == "cuda"):
                loss = model(**model_batch).loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            running_loss += float(loss.item()) * args.gradient_accumulation_steps

            is_update_step = step % args.gradient_accumulation_steps == 0
            is_last_step = step == len(train_loader)
            if is_update_step or is_last_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(loss=running_loss / step)

        metrics = evaluate(
            model,
            tokenizer,
            eval_loader,
            device,
            generation_max_length=args.generation_max_length,
        )
        metrics["epoch"] = epoch
        metrics["train_loss"] = running_loss / max(1, len(train_loader))
        history.append(metrics)
        selection_score = float(metrics[args.selection_metric])
        log.info(
            "epoch=%d valid_accuracy=%.4f balanced_accuracy=%.4f macro_f1=%.4f train_loss=%.4f",
            epoch,
            metrics["accuracy"],
            metrics["balanced_accuracy"],
            metrics["macro_f1"],
            metrics["train_loss"],
        )

        if selection_score > best_score:
            best_score = selection_score
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            with open(args.output_dir / "best_metrics.json", "w", encoding="utf8") as fh:
                json.dump(metrics, fh, indent=2)

        with open(args.output_dir / "history.json", "w", encoding="utf8") as fh:
            json.dump(history, fh, indent=2)

        if not args.no_epoch_checkpoints:
            save_epoch_checkpoint(
                checkpoint_dir=args.output_dir / f"checkpoint_epoch_{epoch}",
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_score=best_score,
                history=history,
                args=args,
            )

    with open(args.output_dir / "label_map.json", "w", encoding="utf8") as fh:
        json.dump(
            {
                "id2label": LABEL_MAP,
                "label2id": LABEL_TO_ID,
                "id2option": ID_TO_OPTION,
                "option2id": OPTION_TO_ID,
            },
            fh,
            indent=2,
        )
    log.info("best valid %s=%.4f saved to %s", args.selection_metric, best_score, args.output_dir)


if __name__ == "__main__":
    main()
