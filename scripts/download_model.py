from __future__ import annotations

from huggingface_hub import constants as hf_constants

from src.classifier import load_classifier
from src.config import CACHE_DIR, MODEL_NAME


def main() -> None:
    tokenizer, model = load_classifier()
    param_count = sum(p.numel() for p in model.parameters())
    device = next(model.parameters()).device
    cache = CACHE_DIR or hf_constants.HF_HUB_CACHE

    print(f"Loaded {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Params: {param_count / 1e6:.1f}M")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Cache: {cache}")


if __name__ == "__main__":
    main()
