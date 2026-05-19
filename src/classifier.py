from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import CACHE_DIR, DEVICE, LABEL_MAP, MODEL_NAME, NUM_LABELS


def load_classifier():
    # The classification head is randomly initialized until fine-tuned;
    # outputs are not meaningful yet — this just exercises the pipeline.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        cache_dir=CACHE_DIR,
    )
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def classify(query: str, tokenizer, model) -> dict:
    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    label_id = int(torch.argmax(logits, dim=-1).item())
    return {
        "label": LABEL_MAP[label_id],
        "label_id": label_id,
        "logits": logits.squeeze(0).tolist(),
    }
