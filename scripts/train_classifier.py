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
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.config import CACHE_DIR, LABEL_MAP, NUM_LABELS


log = logging.getLogger(__name__)

LABEL_TO_ID = {v: k for k, v in LABEL_MAP.items()}


@dataclass(frozen=True)
class Example:
    qid: str
    question: str
    label: int
    dataset: str


class QuestionDataset(Dataset):
    def __init__(self, examples: list[Example], tokenizer: Any, max_length: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        encoded = self.tokenizer(
            ex.question,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(ex.label, dtype=torch.long),
            "qid": ex.qid,
            "dataset": ex.dataset,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune an encoder classifier to route questions to Adaptive-RAG strategies.")
    p.add_argument("--train-file", type=Path, default=Path("data/labeled/classifier_train_binary_silver.jsonl"))
    p.add_argument("--validation-file", type=Path, default=Path("data/labeled/classifier_valid_silver.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/classifier/deberta-v3-large-binary-silver"))
    p.add_argument("--model-name", type=str, default="microsoft/deberta-v3-large")
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=13370)
    p.add_argument("--class-weights", action="store_true", help="Weight loss inversely to class frequency.")
    p.add_argument("--fp16", action="store_true", help="Use CUDA automatic mixed precision.")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Resume from a checkpoint_epoch_N directory written by this script.",
    )
    p.add_argument(
        "--no-epoch-checkpoints",
        action="store_true",
        help="Do not save checkpoint_epoch_N directories after each epoch.",
    )
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
            label_name = row["oracle_label"]
            examples.append(
                Example(
                    qid=str(row["question_id"]),
                    question=row["question_text"],
                    label=LABEL_TO_ID[label_name],
                    dataset=row["dataset"],
                )
            )
            if max_samples is not None and len(examples) >= max_samples:
                break
    return examples


def class_weight_tensor(examples: list[Example], device: torch.device) -> torch.Tensor:
    counts = collections.Counter(ex.label for ex in examples)
    total = sum(counts.values())
    weights = [
        total / (NUM_LABELS * counts[label_id])
        for label_id in range(NUM_LABELS)
    ]
    return torch.tensor(weights, dtype=torch.float, device=device)


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(device),
    }


def evaluate(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    correct = 0
    total = 0
    per_label: dict[str, dict[str, int]] = {
        name: {"correct": 0, "total": 0, "predicted": 0}
        for name in LABEL_MAP.values()
    }
    per_dataset: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"correct": 0, "total": 0})

    with torch.no_grad():
        for batch in dataloader:
            model_batch = batch_to_device(batch, device)
            logits = model(**model_batch).logits
            preds = torch.argmax(logits, dim=-1).cpu()
            labels = batch["labels"]
            matches = preds.eq(labels)

            correct += int(matches.sum().item())
            total += int(labels.numel())

            for pred, label, ok, dataset in zip(preds.tolist(), labels.tolist(), matches.tolist(), batch["dataset"]):
                label_name = LABEL_MAP[label]
                pred_name = LABEL_MAP[pred]
                per_label[label_name]["total"] += 1
                per_label[pred_name]["predicted"] += 1
                per_dataset[dataset]["total"] += 1
                if ok:
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
    metrics["per_dataset"] = {
        dataset: {
            **values,
            "accuracy": values["correct"] / values["total"] if values["total"] else 0.0,
        }
        for dataset, values in sorted(per_dataset.items())
    }
    return metrics


def checkpoint_epoch_from_path(path: Path) -> int:
    try:
        return int(path.name.rsplit("_", 1)[1])
    except (IndexError, ValueError) as e:
        raise ValueError(f"Checkpoint directory must be named checkpoint_epoch_N: {path}") from e


def save_epoch_checkpoint(
    *,
    checkpoint_dir: Path,
    model: AutoModelForSequenceClassification,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_accuracy: float,
    history: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save(
        {
            "epoch": epoch,
            "best_accuracy": best_accuracy,
            "history": history,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
        checkpoint_dir / "training_state.pt",
    )


def load_training_state(
    checkpoint_dir: Path,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> tuple[int, float, list[dict[str, Any]]]:
    state_path = checkpoint_dir / "training_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing training state: {state_path}")
    state = torch.load(state_path, map_location=device)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    return int(state["epoch"]) + 1, float(state.get("best_accuracy", -1.0)), list(state.get("history", []))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                args.output_dir / "train.log",
                mode="a" if args.resume_from_checkpoint else "w",
                encoding="utf8",
            ),
        ],
    )
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_source = args.resume_from_checkpoint or args.model_name
    log.info("device=%s model_source=%s output_dir=%s", device, model_source, args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(model_source), cache_dir=CACHE_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_source),
        num_labels=NUM_LABELS,
        id2label=LABEL_MAP,
        label2id=LABEL_TO_ID,
        cache_dir=CACHE_DIR,
    )
    model.to(device)

    train_examples = load_examples(args.train_file, args.max_train_samples)
    eval_examples = load_examples(args.validation_file, args.max_eval_samples)
    log.info("train=%d valid=%d", len(train_examples), len(eval_examples))
    log.info("train labels=%s", collections.Counter(ex.label for ex in train_examples))
    log.info("valid labels=%s", collections.Counter(ex.label for ex in eval_examples))

    train_dataset = QuestionDataset(train_examples, tokenizer, args.max_length)
    eval_dataset = QuestionDataset(eval_examples, tokenizer, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    update_steps_per_epoch = max(1, (len(train_loader) + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps)
    total_steps = args.epochs * update_steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
    loss_fn = None
    if args.class_weights:
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weight_tensor(train_examples, device))
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    log.info(
        "epochs=%d lr=%g batch_size=%d grad_accum=%d effective_batch_size=%d max_length=%d",
        args.epochs,
        args.learning_rate,
        args.batch_size,
        args.gradient_accumulation_steps,
        effective_batch_size,
        args.max_length,
    )

    best_accuracy = -1.0
    history: list[dict[str, Any]] = []
    start_epoch = 1
    if args.resume_from_checkpoint is not None:
        start_epoch, best_accuracy, history = load_training_state(
            args.resume_from_checkpoint,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        log.info(
            "resumed from %s; next_epoch=%d best_accuracy=%.4f prior_history=%d",
            args.resume_from_checkpoint,
            start_epoch,
            best_accuracy,
            len(history),
        )

    if start_epoch > args.epochs:
        log.info("checkpoint is already past requested epochs (%d > %d); nothing to train", start_epoch, args.epochs)
        return

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(progress, start=1):
            model_batch = batch_to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                outputs = model(**model_batch)
                loss = outputs.loss
                if loss_fn is not None:
                    loss = loss_fn(outputs.logits, model_batch["labels"])
                loss = loss / args.gradient_accumulation_steps

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

        metrics = evaluate(model, eval_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = running_loss / max(1, len(train_loader))
        history.append(metrics)
        log.info("epoch=%d valid_accuracy=%.4f train_loss=%.4f", epoch, metrics["accuracy"], metrics["train_loss"])

        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            with open(args.output_dir / "best_metrics.json", "w", encoding="utf8") as fh:
                json.dump(metrics, fh, indent=2)

        with open(args.output_dir / "history.json", "w", encoding="utf8") as fh:
            json.dump(history, fh, indent=2)

        if not args.no_epoch_checkpoints:
            checkpoint_dir = args.output_dir / f"checkpoint_epoch_{epoch}"
            save_epoch_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_accuracy=best_accuracy,
                history=history,
                args=args,
            )
            log.info("saved checkpoint to %s", checkpoint_dir)

    with open(args.output_dir / "label_map.json", "w", encoding="utf8") as fh:
        json.dump({"id2label": LABEL_MAP, "label2id": LABEL_TO_ID}, fh, indent=2)
    log.info("best valid accuracy=%.4f saved to %s", best_accuracy, args.output_dir)


if __name__ == "__main__":
    main()
