from __future__ import annotations

import pytest
import torch

from src.classifier import classify, load_classifier
from src.config import LABEL_MAP, NUM_LABELS


def test_cuda_available():
    assert torch.cuda.is_available(), "Expected CUDA-capable GPU to be visible to PyTorch"


@pytest.fixture(scope="module")
def loaded():
    tokenizer, model = load_classifier()
    return tokenizer, model


def test_classifier_loads(loaded):
    tokenizer, model = loaded
    assert tokenizer is not None
    assert next(model.parameters()).is_cuda, "Model should be on CUDA"


def test_classify_returns_valid_label(loaded):
    tokenizer, model = loaded
    result = classify("What is the capital of France?", tokenizer, model)
    assert result["label"] in LABEL_MAP.values()
    assert 0 <= result["label_id"] < NUM_LABELS
    assert len(result["logits"]) == NUM_LABELS
