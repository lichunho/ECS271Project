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

## LLM (Ollama)

The answering LLM is served by [Ollama](https://ollama.com/) via its OpenAI-compatible local server (`http://localhost:11434/v1`). Ollama is used (rather than LM Studio) because as of LM Studio 0.3.x the OpenAI-compat layer returns `null` for per-token logprobs — which the Step 5 confidence probe depends on.

1. Install Ollama from [ollama.com/download](https://ollama.com/download). The installer registers a background service; no further start step is needed.
2. Pull a model:
   ```powershell
   ollama pull gemma4:latest   # ~9 GB; or any chat model you prefer
   ```
3. Point `LLM_MODEL` at whatever you pulled (defaults to `gemma4:latest`):
   ```powershell
   $env:LLM_MODEL = "gemma4:latest"
   ```
4. Smoke-test the connection:
   ```powershell
   python -m scripts.llm_smoke
   ```
   You should see a short probe response with a numeric mean logprob, followed by a fuller answer.

The probe and full-answer modes share an identical system prompt and message structure — only `max_tokens` and the `logprobs` flag differ — so the probe's confidence genuinely reflects the answerer's knowledge.

To point the client at a different OpenAI-compatible server (e.g. LM Studio on `:1234`), set `LLM_BASE_URL` and `LLM_API_KEY` in `.env`.

## Roadmap

- [x] Local RoBERTa classifier loads + runs inference on GPU
- [ ] Training data: synthesize A/B/C labels from SQuAD / HotpotQA / MuSiQue
- [ ] Fine-tune the classifier
- [ ] Retriever (BM25 + dense)
- [x] LLM client (probe + answer via Ollama)
- [ ] Confidence-gated routing
- [ ] End-to-end eval harness
