import os

import torch

MODEL_NAME = "FacebookAI/roberta-large"

NUM_LABELS = 3

LABEL_MAP = {
    0: "no_retrieval",
    1: "single_step",
    2: "multi_step",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CACHE_DIR = os.environ.get("HF_HOME")
