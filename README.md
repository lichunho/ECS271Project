# ECS271Project

Adaptive RAG pipeline (in the style of Jeong et al. 2024, *"Adaptive-RAG"*). A small classifier routes incoming queries to one of three retrieval strategies:

- **A** — no retrieval (LLM answers directly)
- **B** — single-step retrieval (one-hop)
- **C** — multi-step retrieval (multi-hop)

This first slice stands up the classifier piece only: download a RoBERTa-large checkpoint and run inference locally on GPU. The classification head is randomly initialized — outputs are not meaningful until fine-tuning lands in a later step.

## Setup (Windows + RTX 50-series / Blackwell)

1. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install PyTorch with CUDA 12.8 wheels (required for Blackwell sm_120):
   ```powershell
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   ```
   If the stable wheel fails on Python 3.13, fall back to the nightly index:
   ```powershell
   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
   ```

3. Install the rest:
   ```powershell
   pip install -r requirements.txt
   ```

4. Verify CUDA visibility:
   ```powershell
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   Expected: `True NVIDIA GeForce RTX 5070 Ti` (or whatever GPU is present).

5. Prefetch model weights (~1.4 GB for `roberta-large`):
   ```powershell
   python -m scripts.download_model
   ```

6. Smoke tests:
   ```powershell
   pytest tests/ -v
   ```

To change where weights are cached, copy `.env.example` to `.env` and set `HF_HOME`, or set the env var directly.

## Roadmap

- [x] Local RoBERTa classifier loads + runs inference on GPU
- [ ] Training data: synthesize A/B/C labels from SQuAD / HotpotQA / MuSiQue
- [ ] Fine-tune the classifier
- [ ] Retriever (BM25 + dense)
- [ ] LLM integration and routing
- [ ] End-to-end eval harness
