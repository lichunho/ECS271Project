import os
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL_NAME = "FacebookAI/roberta-large"

NUM_LABELS = 3

LABEL_MAP = {
    0: "no_retrieval",
    1: "single_step",
    2: "multi_step",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CACHE_DIR = os.environ.get("HF_HOME")

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "ollama")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma4:latest")

PROBE_MAX_TOKENS = 32
ANSWER_MAX_TOKENS = 512
LLM_TEMPERATURE = 0.0
